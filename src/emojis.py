"""Shared happy/sad emoji picks for UI status messages.

Keep this module free of internal imports so widgets and readers can both use it.
Choices are made once at import and reused for subsequent messages.
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

happy_emoji = random.choice(happy_celebratory_emojis)

sad_emoji = random.choice(sad_regretful_emojis)

# Three distinct happy emojis, selected once at import for multi-emoji messages.
three_happy_emojis = "".join(random.sample(happy_celebratory_emojis, 3))
