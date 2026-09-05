"""Responsive layout for the 2D distribution's square plotting area."""
import streamlit as st


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
