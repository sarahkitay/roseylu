"""Curated factual answers for common History, English, and Math questions,
checked before falling back to the generative model.

Same reasoning as app/illustration/topic_responses.py, extended past pure
arithmetic: the from-scratch local model has no reliable general knowledge
(it was trained on ~600K characters of children's literature plus a few
dozen hand-written dialogue examples, not an encyclopedia), so open-ended
factual questions -- "how did Christopher Columbus come to the Americas and
why" -- produced fluent-looking nonsense during live testing. Rather than
keep chasing this with more from-scratch training (three rounds of that
already showed inconsistent, marginal returns -- see training/README.md),
well-known curriculum topics get a hand-written, correct answer looked up by
keyword instead. This is authored content, not text copied from any
textbook -- no copyrighted material is reproduced here.

STATUS: a curated starting set across three subjects, not an exhaustive
K-12 curriculum -- there is no realistic way to hand-author "every textbook
for every grade" in one sitting, and claiming otherwise would be dishonest.
Every answer here is a single, ungraded register (roughly accessible from
age 9-14) -- unlike the redirect_engine templates, these are NOT yet tiered
by AgeTier the way the product's other content is. That's a known
simplification, not an oversight: tiering every topic here 3x would have
tripled the authoring effort for this first pass. Worth doing before this
goes anywhere near a real curriculum claim.

Sensitive history topics (slavery, war) are included because they are
standard elementary/middle-school curriculum, not because they were sought
out -- written at a level intended to be honest without being graphic,
following the same spirit as age-appropriate textbooks, but this content
has NOT been reviewed by a history educator or curriculum specialist,
exactly the same caveat as everywhere else in this repo that touches
sensitive material. Treat it as a draft, not an authority.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QAEntry:
    subject: str
    topic_id: str
    keywords: list[str]
    answer: str


HISTORY: list[QAEntry] = [
    QAEntry(
        "history", "columbus",
        ["christopher columbus", "columbus come", "columbus sail", "columbus discover"],
        "Christopher Columbus was an explorer from Italy who sailed for Spain. In 1492, he "
        "led three ships -- the Nina, the Pinta, and the Santa Maria -- across the Atlantic "
        "Ocean, hoping to find a faster trade route to Asia. Instead, he landed in the "
        "Caribbean, in what was, to Europeans, an entirely unknown part of the world. His "
        "voyages opened the door to widespread European exploration and settlement of the "
        "Americas -- but it's worth knowing that millions of Indigenous people already lived "
        "there, with their own long histories, and this contact eventually led to great harm "
        "to those communities. That's why some people celebrate Columbus as an explorer and "
        "others focus on the impact his arrival had on Indigenous peoples -- both parts are "
        "true and worth thinking about."
    ),
    QAEntry(
        "history", "thanksgiving_pilgrims",
        ["pilgrims", "first thanksgiving", "mayflower"],
        "The Pilgrims were a group of English settlers who sailed to America on a ship called "
        "the Mayflower in 1620, looking for religious freedom. Their first winter was brutal, "
        "and about half of them died. The Wampanoag people, who already lived in that area, "
        "helped the survivors learn to farm local crops and fish. In the fall of 1621, the "
        "Pilgrims and the Wampanoag shared a harvest feast, which is where the story of the "
        "first Thanksgiving comes from."
    ),
    QAEntry(
        "history", "american_revolution",
        ["american revolution", "revolutionary war", "why did the colonists"],
        "The American colonists fought the Revolutionary War (1775-1783) because they felt "
        "Britain was taxing them and controlling their laws without giving them any say -- the "
        "famous phrase was 'no taxation without representation.' The colonies declared their "
        "independence in 1776, and after years of fighting, with help from France, they won, "
        "becoming the United States of America."
    ),
    QAEntry(
        "history", "declaration_of_independence",
        ["declaration of independence"],
        "The Declaration of Independence was adopted on July 4, 1776, and announced that the "
        "13 American colonies considered themselves free from British rule. It was mainly "
        "written by Thomas Jefferson and famously states that people have unalienable rights "
        "to 'life, liberty, and the pursuit of happiness.' That date is why the Fourth of July "
        "is celebrated as America's independence day."
    ),
    QAEntry(
        "history", "george_washington",
        ["george washington"],
        "George Washington led the American army during the Revolutionary War and later became "
        "the first President of the United States in 1789. He's sometimes called the 'Father "
        "of His Country' because of how central he was to founding the nation, and he set an "
        "important example by voluntarily giving up power after two terms instead of ruling "
        "for life."
    ),
    QAEntry(
        "history", "abraham_lincoln_civil_war",
        ["abraham lincoln"],
        "Abraham Lincoln was the 16th President of the United States, leading the country "
        "through the Civil War (1861-1865). He's best known for the Emancipation Proclamation, "
        "which declared enslaved people in Confederate states to be free, and for helping "
        "guide the passage of the 13th Amendment, which abolished slavery everywhere in the "
        "country. He was assassinated in 1865, shortly after the war ended."
    ),
    QAEntry(
        "history", "slavery_civil_war",
        ["civil war", "what caused the civil war", "why did the civil war"],
        "The American Civil War (1861-1865) was fought mainly over slavery -- Southern states "
        "wanted to keep enslaved people as forced, unpaid labor, and wanted to expand slavery "
        "into new territories, while the Northern states increasingly opposed it. When "
        "Abraham Lincoln, who opposed slavery's expansion, was elected president, 11 Southern "
        "states broke away to form the Confederacy rather than accept that. The North won, "
        "the country stayed united, and slavery was abolished -- but it took a long, costly "
        "war for that to happen, and the effects of slavery and the discrimination that "
        "followed it shaped the country for a very long time after."
    ),
    QAEntry(
        "history", "mlk_civil_rights",
        ["martin luther king", "civil rights movement"],
        "Martin Luther King Jr. was a minister and one of the most important leaders of the "
        "Civil Rights Movement, which fought to end racial segregation and unfair treatment of "
        "Black Americans in the 1950s and 60s. He believed strongly in peaceful, nonviolent "
        "protest, and his 'I Have a Dream' speech in 1963 is one of the most famous speeches "
        "in American history. His work helped lead to laws like the Civil Rights Act of 1964, "
        "which made racial discrimination illegal."
    ),
    QAEntry(
        "history", "ancient_egypt",
        ["ancient egypt", "hieroglyphics", "egyptian mummies", "why did egyptians make mummies"],
        "Ancient Egypt was one of the world's earliest great civilizations, built along the "
        "Nile River more than 5,000 years ago. Egyptians believed in an afterlife, which is "
        "why they built huge pyramids as tombs for their pharaohs and made mummies -- "
        "carefully preserving bodies so the person could use them in the afterlife. They also "
        "invented one of the first writing systems, called hieroglyphics, which used pictures "
        "and symbols instead of letters."
    ),
    QAEntry(
        "history", "ancient_greece_rome",
        ["ancient greece", "ancient rome", "roman empire"],
        "Ancient Greece, around 2,500 years ago, gave us early ideas about democracy (rule by "
        "the people), along with philosophy, theater, and the Olympic Games. Ancient Rome grew "
        "from a small city into a massive empire that at one point controlled most of Europe, "
        "spreading Roman law, roads, and architecture everywhere it went. A lot of the "
        "government, buildings, and even words we use today trace back to these two "
        "civilizations."
    ),
    QAEntry(
        "history", "world_war_2",
        ["world war 2", "world war ii", "why did world war 2 start"],
        "World War 2 (1939-1945) was the largest war in history, fought between the Allied "
        "powers (including the US, Britain, and the Soviet Union) and the Axis powers "
        "(mainly Nazi Germany, Italy, and Japan). It started after Germany, led by Adolf "
        "Hitler, invaded Poland, following years of aggression and broken agreements in "
        "Europe. During the war, the Nazi regime murdered six million Jewish people and "
        "millions of others in the Holocaust, one of the worst atrocities in human history. "
        "The war ended in 1945 after the Allies defeated Germany and then Japan."
    ),
    QAEntry(
        "history", "moon_landing",
        ["first person on the moon", "neil armstrong", "moon landing", "apollo 11"],
        "Neil Armstrong was the first person to walk on the moon, on July 20, 1969, during the "
        "Apollo 11 mission. He famously said, 'That's one small step for man, one giant leap "
        "for mankind' as he stepped onto the surface. Buzz Aldrin joined him on the moon a few "
        "minutes later, while Michael Collins stayed in orbit piloting the command module."
    ),
    QAEntry(
        "history", "native_americans",
        ["native americans before", "indigenous people before columbus", "who lived in america before"],
        "Long before any European explorers arrived, the Americas were home to millions of "
        "Indigenous people, organized into hundreds of distinct nations and cultures -- each "
        "with its own language, government, traditions, and ways of living, spread across "
        "North and South America for thousands of years. European colonization brought "
        "devastating disease, warfare, and displacement to these communities, and Indigenous "
        "nations and cultures still exist today across the Americas."
    ),
    QAEntry(
        "history", "branches_of_government",
        ["branches of government", "legislative executive judicial"],
        "The U.S. government is split into three branches so no single part gets too much "
        "power: the Legislative branch (Congress) makes laws, the Executive branch "
        "(the President) enforces laws, and the Judicial branch (the courts, including the "
        "Supreme Court) interprets laws and decides if they're constitutional. This is called "
        "'checks and balances' -- each branch can limit the power of the others."
    ),
]

ENGLISH: list[QAEntry] = [
    QAEntry(
        "english", "parts_of_speech",
        ["what is a noun", "what is a verb", "what is an adjective", "parts of speech"],
        "A noun is a person, place, thing, or idea (like 'dog,' 'school,' or 'happiness'). A "
        "verb is an action or state of being (like 'run,' 'think,' or 'is'). An adjective "
        "describes a noun (like 'fluffy' in 'fluffy dog'). One trick: try putting 'the' in "
        "front of a word -- if it makes sense ('the dog'), it's probably a noun!"
    ),
    QAEntry(
        "english", "simile_metaphor",
        ["difference between a simile and a metaphor", "what's a simile", "what's a metaphor"],
        "Both compare two things, but a simile uses 'like' or 'as' -- like 'brave as a lion.' "
        "A metaphor just says one thing IS another -- like 'he's a lion in a fight.' Want to "
        "try writing one of each about something in your room?"
    ),
    QAEntry(
        "english", "synonym_antonym",
        ["what is a synonym", "what is an antonym"],
        "A synonym is a word that means almost the same thing as another word -- like 'happy' "
        "and 'joyful.' An antonym is a word that means the opposite -- like 'happy' and 'sad.' "
        "Dictionaries and thesauruses (a thesaurus is a book of synonyms) are great for finding "
        "both when you're writing and want to avoid using the same word over and over."
    ),
    QAEntry(
        "english", "topic_sentence_paragraph",
        ["what is a topic sentence", "how do i write a paragraph"],
        "A topic sentence is the first sentence of a paragraph that tells the reader what the "
        "whole paragraph is going to be about -- like a mini preview. Everything else in the "
        "paragraph should support or explain that main idea. A good habit: write your topic "
        "sentence first, then ask yourself 'does every other sentence connect back to this?'"
    ),
    QAEntry(
        "english", "main_idea",
        ["what is the main idea", "how do i find the main idea"],
        "The main idea is the most important point an author is trying to make -- what the "
        "whole piece is really about, underneath all the details. A good way to find it: read "
        "the whole thing, then try to summarize it in one sentence without looking back. "
        "Whatever you naturally focus on in that sentence is probably the main idea."
    ),
    QAEntry(
        "english", "spelling_rules",
        ["spelling rule", "i before e", "silent e rule"],
        "One classic rule is 'i before e, except after c' (like 'believe' but 'receive') -- "
        "though English has plenty of exceptions, so it's a helpful guess, not a guarantee. "
        "Another useful one is the 'silent e' rule: adding an 'e' to the end of a word often "
        "makes the vowel before it say its own name, like 'cap' becoming 'cape.'"
    ),
    QAEntry(
        "english", "alliteration",
        ["what is alliteration"],
        "Alliteration is when several words in a row start with the same sound, like 'Peter "
        "Piper picked a peck of pickled peppers.' Writers use it to make sentences fun to say "
        "out loud and easier to remember. Want to try making up a silly alliterative sentence "
        "with your own name?"
    ),
    QAEntry(
        "english", "rhyme_poetry",
        ["rhyme scheme", "what makes a poem a poem"],
        "A rhyme scheme is the pattern of rhyming words at the ends of lines in a poem, often "
        "written with letters like AABB (where lines 1 and 2 rhyme, then lines 3 and 4 rhyme) "
        "or ABAB (where every other line rhymes). Not all poems need to rhyme at all, though -- "
        "what really makes something a poem is careful attention to rhythm, word choice, and "
        "how it's broken into lines, not just rhyming."
    ),
    QAEntry(
        "english", "theme_of_book",
        ["what is the theme of", "how do i find the theme"],
        "The theme is the deeper message or lesson a story is exploring -- things like "
        "'friendship helps you through hard times' or 'being different isn't a bad thing.' "
        "Unlike the plot (what happens), the theme is what the story is really *about* "
        "underneath the events. A good way to find it: think about what the main character "
        "learns or how they change by the end."
    ),
    QAEntry(
        "english", "prefix_suffix",
        ["what is a prefix", "what is a suffix"],
        "A prefix is added to the beginning of a word to change its meaning, like 'un-' in "
        "'unhappy' (meaning 'not happy'). A suffix is added to the end, like '-ful' in "
        "'helpful' (meaning 'full of help'). Learning common prefixes and suffixes is a great "
        "shortcut for figuring out words you've never seen before."
    ),
    QAEntry(
        "english", "contractions",
        ["what is a contraction", "contractions in english"],
        "A contraction squishes two words together and uses an apostrophe to show where "
        "letters were left out -- like 'do not' becoming 'don't,' or 'they are' becoming "
        "'they're.' They're common in everyday speech and casual writing, but some teachers "
        "ask you to avoid them in very formal writing."
    ),
    QAEntry(
        "english", "punctuation_basics",
        ["when do i use a comma", "punctuation rules"],
        "A period ends a complete statement. A question mark ends a question. An exclamation "
        "point shows strong feeling or emphasis. Commas are trickier -- they're used to "
        "separate items in a list, join two complete thoughts with a word like 'and' or "
        "'but,' or set off extra information in a sentence. When in doubt, try reading your "
        "sentence out loud -- commas often go where you'd naturally pause."
    ),
    QAEntry(
        "english", "compound_words",
        ["what is a compound word"],
        "A compound word is made by joining two smaller words into one new word, like "
        "'sun' + 'flower' = 'sunflower,' or 'basket' + 'ball' = 'basketball.' Sometimes you "
        "can guess a compound word's meaning just by thinking about its two parts."
    ),
    QAEntry(
        "english", "homophones",
        ["what is a homophone", "there their they're", "your you're"],
        "Homophones are words that sound exactly the same but have different spellings and "
        "meanings, like 'there,' 'their,' and 'they're' -- or 'your' and 'you're.' They trip "
        "up a lot of writers! A trick for 'they're/there/their': 'they're' always means 'they "
        "are,' and 'their' always shows ownership (their book) -- if it's neither of those, "
        "it's probably 'there.'"
    ),
]

MATH: list[QAEntry] = [
    QAEntry(
        "math", "decimals",
        ["what is a decimal"],
        "A decimal is a way of writing a number that isn't whole, using a decimal point to "
        "separate the whole part from the fractional part -- like 3.5, which means 3 whole "
        "things plus 5 tenths of another. Money is a decimal you already use all the time: "
        "$3.50 means 3 dollars and 50 cents, which is 50 hundredths of a dollar."
    ),
    QAEntry(
        "math", "percentages",
        ["what is a percentage", "what does percent mean"],
        "Percent means 'out of 100' -- so 50% means 50 out of 100, which is the same as half. "
        "If you got 90% on a test, that means you got 90 out of every 100 points possible. "
        "Percentages, decimals, and fractions are all different ways of showing the same kind "
        "of amount: 50% = 0.5 = 1/2."
    ),
    QAEntry(
        "math", "telling_time",
        ["how do i tell time", "elapsed time"],
        "On an analog clock, the short hand shows the hour and the long hand shows the "
        "minutes -- each number the long hand passes is 5 minutes. For elapsed time (how much "
        "time passed between two clock times), it often helps to count forward in friendly "
        "chunks: get to the next whole hour first, then add the rest."
    ),
    QAEntry(
        "math", "perimeter_vs_area",
        ["difference between perimeter and area", "what is perimeter"],
        "Perimeter is the distance all the way around the outside edge of a shape -- add up "
        "the length of every side. Area is how much space is inside the shape. For a "
        "rectangle that's 4 feet by 3 feet, the perimeter is 4+3+4+3=14 feet, while the area "
        "is 4 times 3 = 12 square feet."
    ),
    QAEntry(
        "math", "negative_numbers",
        ["what is a negative number", "how do negative numbers work"],
        "Negative numbers are less than zero -- think of a thermometer: temperatures can drop "
        "below 0 degrees. On a number line, negative numbers go to the left of zero, and the "
        "further left a number is, the smaller it actually is (-10 is smaller than -1, even "
        "though 10 is bigger than 1)."
    ),
    QAEntry(
        "math", "place_value",
        ["what is place value"],
        "Place value means the position of a digit in a number changes what it's worth. In "
        "the number 352, the 3 is in the hundreds place (worth 300), the 5 is in the tens "
        "place (worth 50), and the 2 is in the ones place (worth 2). The same digit means "
        "something totally different depending on where it sits."
    ),
    QAEntry(
        "math", "rounding_numbers",
        ["how do i round numbers", "rounding rules"],
        "To round a number, look at the digit right after the place you're rounding to. If "
        "it's 5 or higher, round up; if it's 4 or lower, round down. For example, rounding 47 "
        "to the nearest ten: look at the 7, which is 5 or higher, so it rounds up to 50."
    ),
    QAEntry(
        "math", "even_odd_numbers",
        ["what makes a number even", "what makes a number odd"],
        "A number is even if it can be split into two equal groups with nothing left over -- "
        "it always ends in 0, 2, 4, 6, or 8. A number is odd if one is left over when you try "
        "to split it evenly -- it always ends in 1, 3, 5, 7, or 9."
    ),
    QAEntry(
        "math", "factors_multiples",
        ["what is a factor", "what is a multiple"],
        "A factor is a number that divides evenly into another number -- the factors of 12 "
        "are 1, 2, 3, 4, 6, and 12, since each of those divides into 12 with nothing left "
        "over. A multiple is what you get by multiplying a number by whole numbers -- the "
        "multiples of 4 are 4, 8, 12, 16, and so on."
    ),
    QAEntry(
        "math", "angles_types",
        ["types of angles", "what is an acute angle", "what is an obtuse angle"],
        "A right angle is exactly 90 degrees, like the corner of a square -- you can check for "
        "one with the corner of a piece of paper. An acute angle is smaller than that (a "
        "sharp, narrow angle), and an obtuse angle is bigger than a right angle but still less "
        "than a straight line."
    ),
    QAEntry(
        "math", "probability_basic",
        ["what is probability"],
        "Probability is how likely something is to happen, usually written as a fraction, "
        "decimal, or percent. If you flip a coin, the probability of getting heads is 1 out "
        "of 2 (1/2, or 50%), because there are 2 equally likely outcomes and heads is 1 of "
        "them."
    ),
    QAEntry(
        "math", "reading_graphs",
        ["how do i read a graph", "how do i read a bar chart"],
        "Start with the title, which tells you what the graph is about, then check the "
        "labels on each axis (the lines along the bottom and side) to see what's being "
        "measured. For a bar graph, taller or longer bars mean bigger amounts -- compare the "
        "bars to each other to see patterns."
    ),
    QAEntry(
        "math", "algebra_variables",
        ["what is a variable", "what is algebra"],
        "Algebra uses letters, called variables, to stand in for numbers we don't know yet -- "
        "like x in 'x + 3 = 7.' To solve it, you figure out what number x has to be to make "
        "the equation true (here, x = 4). It's basically a puzzle where the letter is the "
        "missing piece."
    ),
    QAEntry(
        "math", "order_of_operations",
        ["order of operations", "pemdas"],
        "Order of operations is the agreed-upon order for solving a math problem with "
        "multiple steps, often remembered as PEMDAS: Parentheses, Exponents, Multiplication "
        "and Division (left to right), then Addition and Subtraction (left to right). Without "
        "an agreed order, the same problem could give different answers to different people."
    ),
    QAEntry(
        "math", "measurement_conversion",
        ["how many inches in a foot", "measurement conversion", "how many feet in a mile"],
        "Some common ones worth memorizing: 12 inches = 1 foot, 3 feet = 1 yard, and "
        "5,280 feet = 1 mile. For metric, it's cleaner: 100 centimeters = 1 meter, and "
        "1,000 meters = 1 kilometer -- metric conversions are all powers of 10, which is why "
        "many scientists prefer it."
    ),
    QAEntry(
        "math", "pythagorean_theorem",
        # "pythag" (not "pythagorean") on purpose -- it's a substring of common
        # misspellings too, like "pythagrean," which is exactly the phrasing
        # that first surfaced this gap during live testing.
        ["pythag"],
        "The Pythagorean theorem works for right triangles (triangles with one 90-degree "
        "corner): if you square the two shorter sides and add them together, you get the "
        "square of the longest side (the hypotenuse, the one opposite the right angle). "
        "Written as a formula: a^2 + b^2 = c^2. So if the two shorter sides are 3 and 4, "
        "3^2 + 4^2 = 9 + 16 = 25, and the square root of 25 is 5 -- the hypotenuse is 5."
    ),
    QAEntry(
        "math", "square_roots",
        ["what is a square root", "how do square roots work"],
        "A square root asks 'what number, multiplied by itself, gives me this?' The square "
        "root of 9 is 3, because 3 times 3 is 9. The little checkmark-shaped symbol (√) means "
        "'square root of.' Perfect squares like 4, 9, 16, and 25 have whole-number square "
        "roots (2, 3, 4, and 5) -- most other numbers don't."
    ),
    QAEntry(
        "math", "exponents",
        ["what is an exponent", "what does squared mean", "what does cubed mean"],
        "An exponent tells you how many times to multiply a number by itself. 3^2 (3 "
        "'squared') means 3 times 3, which is 9. 3^3 (3 'cubed') means 3 times 3 times 3, "
        "which is 27. The little raised number is the exponent, and the number underneath "
        "it is called the base."
    ),
    # Basic shapes: added after live testing showed the local model
    # confidently retrieving the WRONG shape's fact -- "how many sides does
    # a square have" got answered with "3 sides" (a triangle fact it had
    # memorized from a similar-sounding training example). That's a
    # different, more concerning failure mode than incoherent text: a
    # fluent, confident, factually WRONG answer. Deterministic per-shape
    # answers close it the same way the other curated topics do.
    QAEntry(
        "math", "square_shape",
        ["sides does a square", "square shape", "square have"],
        "A square has exactly 4 sides, all the same length, and 4 corners that are all "
        "right angles (90 degrees). If a shape has 4 equal sides but its corners AREN'T "
        "right angles, it's a rhombus, not a square."
    ),
    QAEntry(
        "math", "triangle_shape",
        ["sides does a triangle", "triangle have"],
        "A triangle always has exactly 3 sides and 3 corners, no matter how big, small, "
        "pointy, or wide it looks. The three angles inside a triangle always add up to "
        "180 degrees, no matter the triangle's shape."
    ),
    QAEntry(
        "math", "rectangle_shape",
        ["sides does a rectangle", "rectangle have", "what is a rectangle"],
        "A rectangle has 4 sides and 4 right-angle corners, like a square -- but unlike a "
        "square, a rectangle's sides don't all have to be the same length. It just needs "
        "each pair of opposite sides to be equal (two long sides matching each other, two "
        "short sides matching each other)."
    ),
    QAEntry(
        "math", "circle_shape",
        ["what is a circle", "sides does a circle"],
        "A circle has no straight sides or corners at all -- it's one continuous curved "
        "line where every point is exactly the same distance from the center. That "
        "distance from the center to the edge is called the radius."
    ),
    # "What is [advanced math branch]" cluster -- added together after
    # "how does calculus work" surfaced the same gap pattern as everything
    # else in this file: a curious kid asking about a subject by name, well
    # past what synthetic_dialogues.py happens to cover, producing
    # incoherent text. Kids curious enough to ask about calculus tend to ask
    # about its neighbors too, so covering the cluster now beats fixing one
    # more of these next week.
    QAEntry(
        "math", "calculus",
        ["how does calculus work", "what is calculus"],
        "Calculus is the math of change and accumulation. One half, called derivatives, "
        "tells you how fast something is changing at a single instant -- like your car's "
        "speedometer showing your exact speed right now, not just your average speed for "
        "the whole trip. The other half, called integrals, adds up tiny pieces to find a "
        "total -- like figuring out the exact area under a curvy line instead of a simple "
        "shape. It's usually taught in high school or college, built on algebra and "
        "geometry you learn first."
    ),
    QAEntry(
        "math", "trigonometry",
        ["what is trigonometry", "how does trigonometry work"],
        "Trigonometry studies the relationships between the angles and sides of triangles, "
        "especially right triangles. It's built around three key ratios -- sine, cosine, "
        "and tangent -- that let you figure out a missing side or angle if you know enough "
        "about the others. It's used a lot in fields like architecture, navigation, and "
        "video game graphics, anywhere you need to calculate angles and distances precisely."
    ),
    QAEntry(
        "math", "statistics",
        ["what is statistics", "what does statistics mean"],
        "Statistics is the math of collecting, organizing, and making sense of data -- "
        "numbers and information about the real world. It helps answer questions like "
        "'what's typical?' (using averages), 'how spread out are these results?', or 'how "
        "confident can we be in this conclusion?' Every time you see a poll, a batting "
        "average, or a weather forecast's chance of rain, that's statistics at work."
    ),
]

ALL_ENTRIES: list[QAEntry] = HISTORY + ENGLISH + MATH


def find_answer(message: str) -> str | None:
    text = message.lower()
    for entry in ALL_ENTRIES:
        if any(kw in text for kw in entry.keywords):
            return entry.answer
    return None
