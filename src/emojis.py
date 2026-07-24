"""Per-session happy/sad emoji picks used to sign status messages across the UI.

Deliberately free of internal imports so any module can import it. These lists used
to live in ``src.dataset_io``, which imports ``src.widgets.analysis_config_widgets``
and therefore could not be imported back from the widget modules that need them.

The ``random.choice`` calls run once at import, so a session shows one consistent
happy emoji and one consistent sad emoji rather than a new face on every message.
"""

import random

happy_celebratory_emojis = [
    "🥳",  # Partying Face
    "🎉",  # Party Popper
    "🎊",  # Confetti Ball
    "✨",  # Sparkles
    "🎈",  # Balloon
    "🎆",  # Fireworks
    "🎇",  # Sparkler
    "🤩",  # Star-Struck
    "😊",  # Smiling Face with Smiling Eyes
    "😃",  # Grinning Face with Big Eyes
    "😁",  # Beaming Face with Smiling Eyes
    "😄",  # Grinning Face with Smiling Eyes
    "🥰",  # Smiling Face with Hearts
    "🙌",  # Raising Hands
    "🥂",  # Clinking Glasses
    "🍾",  # Bottle with Popping Cork
    "👍",  # Thumbs Up
    "😉",
    "💛", 
    "🩵",
    "🍰",  # Shortcake
    "🌟",  # Glowing Star
    "💃",  # Woman Dancing
    "🕺",  # Man Dancing
    "🚀",  # Rocket
    "🌈",  # Rainbow
    "🦄",  # Unicorn
    "🎨",  # Artist Palette
    "🏆",  # Trophy
    "🏅",  # Sports Medal
    "🎯",  # Bullseye
    "🎡",  # Ferris Wheel
    "🧸",  # Teddy Bear
    "😸",  # Grinning Cat with Smiling Eyes
    "🛝",  # Slide
    "🎠",  # Carousel Horse
    "🎢",  # Roller Coaster
    "🪁",  # Kite
    "🎪",  # Circus Tent
    "🤹",  # Person Juggling
    "🤸",  # Person Cartwheeling
    "🛹",  # Skateboard
    "🛴",  # Kick Scooter
    "🥟",  # Dumpling
    "🍕",  # Pizza
    "🍔",  # Hamburger
    "🍟",  # French Fries
    "🌮",  # Taco
    "🍿",  # Popcorn
    "🎂",  # Birthday Cake
    "🧁",  # Cupcake
    "🍩",  # Doughnut
    "🍦",  # Soft Ice Cream
    "🍨",  # Ice Cream
    "🥞",  # Pancakes
    "🧇",  # Waffle
    "🍬",  # Candy
    "🍭",  # Lollipop
    "🍫",  # Chocolate Bar
    "🍓",  # Strawberry
    "🍒",  # Cherries
    "🍉",  # Watermelon
    "🧋",  # Bubble Tea
    "🎡",  # Ferris Wheel
    "🧸",  # Teddy Bear
]

sad_regretful_emojis = [
    "😥",  # Sad but Relieved Face
    "😢",  # Crying Face
    "😭",  # Loudly Crying Face
    "😞",  # Disappointed Face
    "😟",  # Worried Face
    "🥺",  # Pleading Face (can imply regret or sadness)
    "💔",  # Broken Heart
    "😔",  # Pensive Face (can imply contemplation after a mistake)
    "😬",
    "😮‍💨",
    "😶‍🌫️",
    "🤔",
    "🤒",
    "🥶",
    "😖",
    "😫",
    "😩",  # Weary Face
    "☹️",  # Frowning Face
    "🙁",  # Slightly Frowning Face
    "😿",  # Crying Cat
    "😓",  # Downcast Face with Sweat
    "😰",  # Anxious Face with Sweat
    "🫠",  # Melting Face
    "🥀",  # Wilted Flower
    "🌫️",  # Fog
    "📉",  # Chart Decreasing
    "🤕",  # Face with Head-Bandage
    "🥵",  # Hot Face
    "😵‍💫", # Face with Spiral Eyes
    "🤦",  # Person Facepalming
    "😾",  # Pouting Cat
    "😤",  # Face with Steam from Nose
    "💀",  # Skull
    "🧟",  # Zombie
    "💥",  # Collision
    "🪫",  # Low Battery
    "🌪️",  # Tornado
    "🧯",  # Fire Extinguisher
    "🤯",  # Exploding Head
    "🫥",  # Dotted Line Face
    "🫨",  # Shaking Face
    "😶",  # Face Without Mouth
    "🤐",  # Zipper-Mouth Face
    "🥱",  # Yawning Face
    "🙀",  # Weary Cat
    "🌧️",  # Cloud with Rain
    "⛈️",  # Cloud with Lightning and Rain
    "😵",  # Dizzy Face
    "🆘"   # SOS Button
]

# Choose a random happy/celebratory emoji
happy_emoji = random.choice(happy_celebratory_emojis)

# Choose a random sad/regretful emoji
sad_emoji = random.choice(sad_regretful_emojis)

# Three distinct happy emojis for festive multi-emoji spots (e.g. the welcome banner).
# random.sample => no repeats; picked once at import so the trio is stable per session.
three_happy_emojis = "".join(random.sample(happy_celebratory_emojis, 3))
