"""Responsive layout for the 2D distribution's square plotting area."""
import streamlit as st


def phasor_chart(fig, *, key):
    """Size the full-width Phasor category view using native chart margins.

    Plotly's scaleanchor/constrain settings retain physical G/S aspect in
    fullscreen. Normal chart height follows the available plotting width.
    """
    meta = fig.layout.meta
    if not isinstance(meta, dict) or not meta.get("phasor_subplot_layout"):
        st.plotly_chart(fig, width="stretch", key=key)
        return
    st.html("""
        <style>
        .st-key-phasor_subplot_plot [data-testid="stElementContainer"] { min-height: 0; }
        </style>
        <script>
        (() => {
            window._flimPhasorCleanup?.();
            let graph, observedRoot, frame;
            let mounted = false, disposed = false;
            const update = () => {
                const root = document.querySelector('.st-key-phasor_subplot_plot');
                if (!root && mounted) { cleanup(); return; }
                mounted ||= !!root;
                const next = root?.querySelector('.js-plotly-plot');
                if (next !== graph || root !== observedRoot) {
                    graph?.removeListener?.('plotly_afterplot', schedule);
                    resizeObserver.disconnect();
                    graph = next?.on ? next : undefined;
                    observedRoot = root;
                    graph?.on('plotly_afterplot', schedule);
                    if (root) resizeObserver.observe(root);
                    if (graph) {
                        resizeObserver.observe(graph);
                        const fullscreen = graph.closest('[data-testid="stFullScreenFrame"]');
                        if (fullscreen) resizeObserver.observe(fullscreen);
                    }
                }
                const layout = graph?._fullLayout;
                const ratio = layout?.meta?.phasor_subplot_layout?.plot_height;
                if (!layout?._size || !Number.isFinite(ratio) || ratio <= 0) return;
                const fullscreen = graph.closest('[data-testid="stFullScreenFrame"]');
                if (fullscreen && getComputedStyle(fullscreen).position === 'fixed') return;
                const {w, t, b} = layout._size;
                if (w <= 0) return;
                const container = graph.closest('[data-testid="stElementContainer"]');
                const height = `${Math.round(w * ratio + t + b)}px`;
                if (container && container.style.height !== height) container.style.height = height;
            };
            const schedule = () => {
                if (disposed) return;
                cancelAnimationFrame(frame);
                frame = requestAnimationFrame(update);
            };
            const resizeObserver = new ResizeObserver(schedule);
            const observer = new MutationObserver(schedule);
            const cleanup = () => {
                disposed = true;
                observer.disconnect();
                resizeObserver.disconnect();
                cancelAnimationFrame(frame);
                graph?.removeListener?.('plotly_afterplot', schedule);
                if (window._flimPhasorCleanup === cleanup) delete window._flimPhasorCleanup;
            };
            observer.observe(document.body, {childList: true, subtree: true});
            window._flimPhasorCleanup = cleanup;
            schedule();
        })();
        </script>
    """, unsafe_allow_javascript=True)
    with st.container(key="phasor_subplot_plot"):
        st.plotly_chart(fig, width="stretch", height="stretch", key=key)


def square_2d_plot(fig, *, key):
    """Keep the main axes square after resizing, styling, and legend changes.

    Streamlit stretches chart width independently of height. Plotly computes its
    label/legend margins in the browser, so size the native chart there using the
    rendered margins. The 2D distribution's X/Y domains have equal spans.
    """
    st.html("""
        <style>
        .st-key-square_2d_distribution [data-testid="stElementContainer"] {
            min-height: 0;
        }
        </style>
        <script>
        (() => {
            window._flimSquare2dCleanup?.();
            let graph;
            let frame;
            let mounted = false;
            let disposed = false;

            const update = () => {
                const root = document.querySelector('.st-key-square_2d_distribution');
                if (!root && mounted) {
                    cleanup();
                    return;
                }
                mounted ||= !!root;
                const next = root?.querySelector('.js-plotly-plot');
                if (next !== graph) {
                    graph?.removeListener?.('plotly_afterplot', schedule);
                    graph = next?.on ? next : undefined;
                    graph?.on('plotly_afterplot', schedule);
                }
                const layout = graph?._fullLayout;
                if (!layout?._size) return;
                const {l, r, t, b, w, h} = layout._size;
                const fullscreen = graph.closest('[data-testid="stFullScreenFrame"]');
                const expanded = getComputedStyle(fullscreen).position === 'fixed';
                if (!expanded) {
                    const height = Math.round(layout.width - l - r + t + b);
                    const container = graph.closest('[data-testid="stElementContainer"]');
                    const target = `${height}px`;
                    if (container.style.height !== target) container.style.height = target;
                }

                // Fullscreen fixes the canvas to the viewport. Fit both the main
                // axes and their marginals into a centered square inside it.
                // Restore the full domains on exit; data ranges remain independent.
                const changes = {};
                for (const [axis, span] of [
                    ['xaxis', expanded ? Math.min(1, h / w) : 1],
                    ['yaxis', expanded ? Math.min(1, w / h) : 1],
                ]) {
                    const start = (1 - span) / 2;
                    const mainAxes = axis === 'yaxis' ? [axis, 'yaxis3'] : [axis];
                    for (const [name, domain] of [
                        ...mainAxes.map(name => [name, [start, start + 0.9 * span]]),
                        [axis + '2', [start + 0.9 * span, start + span]],
                    ]) {
                        // Constant/small groups can leave the marginal axes unused.
                        if (!layout[name]) continue;
                        if (domain.some((value, i) => Math.abs(value - layout[name].domain[i]) > 1e-6)) {
                            changes[name + '.domain'] = domain;
                        }
                    }
                }
                if (Object.keys(changes).length) window.Plotly.relayout(graph, changes);
            };
            const schedule = () => {
                if (disposed) return;
                cancelAnimationFrame(frame);
                frame = requestAnimationFrame(update);
            };
            const observer = new MutationObserver(schedule);
            const cleanup = () => {
                disposed = true;
                observer.disconnect();
                cancelAnimationFrame(frame);
                graph?.removeListener?.('plotly_afterplot', schedule);
                if (window._flimSquare2dCleanup === cleanup) {
                    delete window._flimSquare2dCleanup;
                }
            };
            observer.observe(document.body, {childList: true, subtree: true});
            window._flimSquare2dCleanup = cleanup;
            schedule();
        })();
        </script>
    """, unsafe_allow_javascript=True)
    with st.container(key="square_2d_distribution"):
        st.plotly_chart(fig, width="stretch", height="stretch", key=key)


def dimension_reduction_chart(fig, *, key):
    """Fit canonical DR panel geometry using Plotly's measured browser margins.

    Normal charts fit the available width and most of the viewport height, with
    room for measured label and legend margins. Fullscreen charts keep the
    viewport size and center the same composition inside it.
    Canonical domains and paper annotations stay in metadata so repeated resizing
    cannot accumulate coordinate drift or alter the user's zoom ranges.
    """
    meta = fig.layout.meta
    if not isinstance(meta, dict) or not meta.get("dimension_reduction_layout"):
        st.plotly_chart(fig, width="stretch", key=key)
        return

    st.html("""
        <style>
        .st-key-dimension_reduction_plot [data-testid="stElementContainer"] {
            min-height: 0;
        }
        </style>
        <script>
        (() => {
            window._flimDimensionReductionCleanup?.();
            let graph;
            let observedRoot;
            let frame;
            let mounted = false;
            let disposed = false;

            const update = () => {
                const root = document.querySelector('.st-key-dimension_reduction_plot');
                if (!root && mounted) {
                    cleanup();
                    return;
                }
                mounted ||= !!root;
                const next = root?.querySelector('.js-plotly-plot');
                if (next !== graph || root !== observedRoot) {
                    graph?.removeListener?.('plotly_afterplot', schedule);
                    resizeObserver.disconnect();
                    graph = next?.on ? next : undefined;
                    observedRoot = root;
                    graph?.on('plotly_afterplot', schedule);
                    if (root) resizeObserver.observe(root);
                    if (graph) {
                        resizeObserver.observe(graph);
                        const fullscreen = graph.closest('[data-testid="stFullScreenFrame"]');
                        if (fullscreen) resizeObserver.observe(fullscreen);
                    }
                }
                const layout = graph?._fullLayout;
                const canonical = layout?.meta?.dimension_reduction_layout;
                const ratio = canonical?.plot_height;
                if (!layout?._size || !Number.isFinite(ratio) || ratio <= 0) return;
                const {w, h, l, r, t, b} = layout._size;
                if (w <= 0 || h <= 0) return;
                const fullscreen = graph.closest('[data-testid="stFullScreenFrame"]');
                const expanded = fullscreen && getComputedStyle(fullscreen).position === 'fixed';
                if (!expanded) {
                    const container = graph.closest('[data-testid="stElementContainer"]');
                    // Read the full parent width so a previously capped chart can
                    // grow again. A height-only viewport resize also matters.
                    const availableInnerWidth = Math.max(1, root.getBoundingClientRect().width - l - r);
                    const heightCap = Math.max(300, Math.round(window.innerHeight * 0.85));
                    // Large legends may need more than the height budget; keep
                    // the map useful rather than shrinking it to a few pixels.
                    const innerHeight = Math.min(availableInnerWidth * ratio,
                        Math.max(120, heightCap - t - b));
                    const targetWidth = `${Math.round(innerHeight / ratio + l + r)}px`;
                    const targetHeight = `${Math.round(innerHeight + t + b)}px`;
                    if (container) {
                        if (container.style.width !== targetWidth) container.style.width = targetWidth;
                        if (container.style.height !== targetHeight) container.style.height = targetHeight;
                    }
                }

                const sx = expanded ? Math.min(1, h / (w * ratio)) : 1;
                const sy = expanded ? Math.min(1, w * ratio / h) : 1;
                const changes = {};
                const transform = (value, span) => (1 - span) / 2 + value * span;
                const differs = (value, current) => !Number.isFinite(current)
                    || Math.abs(value - current) > 1e-6;
                for (const [axis, base] of Object.entries(canonical.axes || {})) {
                    if (!layout[axis]?.domain) continue;
                    const span = axis.startsWith('x') ? sx : sy;
                    const domain = base.map(value => transform(value, span));
                    if (domain.some((value, i) => differs(value, layout[axis].domain[i]))) {
                        changes[axis + '.domain'] = domain;
                    }
                }
                for (const [index, base] of (canonical.annotations || []).entries()) {
                    const current = layout.annotations?.[index];
                    if (!current) continue;
                    for (const [coordinate, span] of [['x', sx], ['y', sy]]) {
                        // Coordinates referenced to an axis domain already follow
                        // that domain and must not be transformed a second time.
                        if (base[coordinate + 'ref'] !== 'paper'
                            || !Number.isFinite(base[coordinate])) continue;
                        const value = transform(base[coordinate], span);
                        if (differs(value, current[coordinate])) {
                            changes[`annotations[${index}].${coordinate}`] = value;
                        }
                    }
                }
                const xAxes = Object.keys(canonical.axes || {}).filter(axis => axis.startsWith('x'));
                // Measure only after the domains/annotations reach their target
                // positions, so entering fullscreen cannot measure stale labels.
                if (xAxes.length > 1 && !Object.keys(changes).length) {
                    const labels = [...graph.querySelectorAll('.infolayer .annotation')]
                        .map(node => node.getBoundingClientRect()).filter(box => box.width > 0);
                    if (labels.length) {
                        const rightDomain = Math.max(...xAxes.map(axis => layout[axis].domain[1]));
                        const rightEdge = graph.getBoundingClientRect().left
                            + layout._size.l + w * rightDomain;
                        const required = Math.max(12,
                            Math.ceil(Math.max(...labels.map(box => box.right)) - rightEdge + 12));
                        // Fullscreen may leave an intentional aspect-ratio gutter.
                        // Reserve label overflow from the composition edge, not
                        // the inner canvas edge, and retain other auto margins.
                        if (Math.abs(required - layout.margin.r) > 1) changes['margin.r'] = required;
                    }
                }
                if (Object.keys(changes).length) window.Plotly.relayout(graph, changes);
            };
            const schedule = () => {
                if (disposed) return;
                cancelAnimationFrame(frame);
                frame = requestAnimationFrame(update);
            };
            const resizeObserver = new ResizeObserver(schedule);
            const observer = new MutationObserver(schedule);
            const cleanup = () => {
                disposed = true;
                observer.disconnect();
                resizeObserver.disconnect();
                window.removeEventListener('resize', schedule);
                cancelAnimationFrame(frame);
                graph?.removeListener?.('plotly_afterplot', schedule);
                if (window._flimDimensionReductionCleanup === cleanup) {
                    delete window._flimDimensionReductionCleanup;
                }
            };
            window.addEventListener('resize', schedule);
            observer.observe(document.body, {childList: true, subtree: true});
            window._flimDimensionReductionCleanup = cleanup;
            schedule();
        })();
        </script>
    """, unsafe_allow_javascript=True)
    with st.container(key="dimension_reduction_plot"):
        st.plotly_chart(fig, width="stretch", height="stretch", key=key)
