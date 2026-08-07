"""Curated factual answers for common History, English, and Math questions,
plus a small set of everyday emotional/family questions (see LIFE below) and
basic science facts (see SCIENCE below), checked before falling back to the
generative model.

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

import difflib
import re
from dataclasses import dataclass, field

from app.config import AgeTier


@dataclass
class QAEntry:
    subject: str
    topic_id: str
    keywords: list[str]
    answer: str  # used as-is when answers_by_tier is empty, and as the fallback for any tier missing from it
    answers_by_tier: dict[AgeTier, str] = field(default_factory=dict)
    # Most entries (still) use a single ungraded `answer` -- see the module
    # docstring's honest accounting of why tiering everything wasn't done in
    # one pass. `answers_by_tier` is opt-in per entry; Columbus is the first
    # to use it, as the flagship example of what full tiering looks like.
    fuzzy_eligible: bool = True
    # False for the LIFE category: _fuzzy_find_answer's anchor heuristic
    # (longest word in a keyword phrase = the word worth typo-tolerating)
    # holds for rare curriculum vocabulary ("calculus", "pythagorean") but
    # breaks for everyday emotional phrasing, where the longest word is
    # often just a common feeling word ("friend", "scared") that a totally
    # unrelated message could contain verbatim -- an EXACT match (ratio
    # 1.0), not a typo, hijacking an unrelated message into the wrong canned
    # answer. Found live via a test regression on "friend". Typo tolerance
    # has low value for casual phrasing anyway (a misspelled "mommy" is a
    # minor miss), while a false-positive emotional-topic mismatch is a
    # worse outcome here than in any other category, so LIFE opts out of
    # fuzzy matching entirely rather than trying to hand-tune anchors safe.

    def answer_for(self, tier: AgeTier | None) -> str:
        if tier is not None and tier in self.answers_by_tier:
            return self.answers_by_tier[tier]
        return self.answer


HISTORY: list[QAEntry] = [
    QAEntry(
        "history", "columbus",
        ["christopher columbus", "columbus come", "columbus sail", "columbus discover", "columbus get to america"],
        # `answer` (below) stays as the MIDDLE-equivalent default for any
        # caller that doesn't pass a tier (e.g. training/scripts/ tooling).
        # `answers_by_tier` is the real, tiered content -- see it for the
        # actual per-age story used by the live app.
        "Picture this: after 10 whole weeks sailing across open ocean with no land in sight, "
        "Christopher Columbus and his crew finally spotted land on October 12, 1492. He'd "
        "sailed from Spain with three ships -- the Nina, the Pinta, and the Santa Maria -- "
        "trying to find a faster trade route to Asia by going west instead of the usual route "
        "around Africa. When he landed on an island in the Caribbean, he named it San Salvador "
        "('Holy Savior') and claimed it for Spain. Here's the twist that changed history: "
        "Columbus was completely convinced he'd landed near India. He hadn't -- he'd stumbled "
        "onto a continent no European had ever known existed -- but because he believed it, he "
        "called the people he met there 'Indians.' That mistaken name ended up sticking for "
        "Indigenous peoples across the Americas for hundreds of years. His voyages opened the "
        "door to massive European exploration and settlement, but millions of Indigenous "
        "people already lived there, with their own rich histories going back thousands of "
        "years -- and what came after brought devastating harm to those communities. "
        "Historians still argue about how to tell this story because of that. What do you "
        "think matters more when we remember Columbus -- the incredible journey, or what "
        "happened to the people who were already there?",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "A long, long time ago, a sailor named Christopher Columbus sailed 3 little "
                "boats across a HUGE ocean. He sailed for many, many weeks! Finally, he found "
                "land. But guess what -- people already lived there! Columbus thought he'd "
                "found a place called India, so he called the people 'Indians' -- even though "
                "he was somewhere totally new and different. Silly mix-up, right? If YOU found "
                "a brand new place, what would you want to name it?"
            ),
            AgeTier.EARLY: (
                "Get this: Christopher Columbus sailed across the ocean for 10 whole weeks -- "
                "that's more than two months! -- with three ships called the Nina, the Pinta, "
                "and the Santa Maria. On October 12, 1492, they finally saw land. Columbus "
                "named the island San Salvador and figured he must have reached India, so he "
                "called the people who already lived there 'Indians.' He was wrong -- he'd "
                "actually found a whole continent nobody in Europe knew existed! -- but that "
                "name stuck around for a really long time. Lots of people already lived in the "
                "Americas before Columbus ever showed up, with their own families, languages, "
                "and traditions. If you sailed for 10 weeks and finally saw land, what's the "
                "first thing you think you'd do?"
            ),
            AgeTier.MIDDLE: (
                "Picture this: after 10 whole weeks sailing across open ocean with no land in "
                "sight, Christopher Columbus and his crew finally spotted land on October 12, "
                "1492. He'd sailed from Spain with three ships -- the Nina, the Pinta, and the "
                "Santa Maria -- trying to find a faster trade route to Asia by going west "
                "instead of the usual route around Africa. When he landed on an island in the "
                "Caribbean, he named it San Salvador ('Holy Savior') and claimed it for Spain. "
                "Here's the twist that changed history: Columbus was completely convinced he'd "
                "landed near India. He hadn't -- he'd stumbled onto a continent no European had "
                "ever known existed -- but because he believed it, he called the people he met "
                "there 'Indians.' That mistaken name ended up sticking for Indigenous peoples "
                "across the Americas for hundreds of years. His voyages opened the door to "
                "massive European exploration and settlement, but millions of Indigenous people "
                "already lived there, with their own rich histories going back thousands of "
                "years -- and what came after brought devastating harm to those communities. "
                "So here's a real question historians still argue about: when we tell this "
                "story, what should we lead with -- the incredible journey, or what happened "
                "to the people who were already there? What's your take?"
            ),
            AgeTier.TEEN: (
                "Here's the version most people don't get in school: Columbus sailed west for "
                "10 weeks in 1492 -- three ships, the Nina, the Pinta, and the Santa Maria -- "
                "trying to find a cheaper route to Asian trade markets. When he hit land in the "
                "Caribbean, he was so sure he'd reached the Indies that he called the people he "
                "met 'Indians,' a mistaken label that outlived him by centuries. He never "
                "actually figured out he'd found an entire continent nobody in Europe knew "
                "existed. That 'discovery' framing is exactly what's contested: for millions of "
                "Indigenous people who already lived there, with their own nations and "
                "histories going back thousands of years, 1492 marks the start of colonization, "
                "disease, and violence, not a triumphant arrival. Both the navigational "
                "achievement and the harm that followed are historically real -- the "
                "disagreement is about which one gets to define how we remember him. Where do "
                "you land on that?"
            ),
        },
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

# Everyday emotional/family questions -- distinct from History/English/Math
# above (this isn't curriculum), added after a live report: "why does mommy
# yell" isn't caught by the guardrail pipeline (it isn't dangerous -- no
# reason to redirect it), isn't curriculum, and has no numeric template, so
# it fell straight through to the generative model, which returned
# completely unrelated square-root/synonym text with zero connection to what
# was actually asked. That's a worse failure than a boring answer: a young
# child asking something emotionally real about their family deserves an
# actually-relevant reply, not word salad. Tiered like Columbus, since the
# right way to talk to a 4-year-old about a parent yelling is genuinely
# different from how you'd talk to a teenager about it, not just simpler
# words for the same paragraph. This is explicitly a starting point, not
# emotional-support coverage -- see the module docstring's honesty about
# curated_qa.py being a curated set, not exhaustive. Rosey isn't a
# therapist and doesn't claim to be; every tier here validates the child's
# feelings and points toward a trusted adult rather than diagnosing what's
# actually happening in their family.
LIFE: list[QAEntry] = [
    QAEntry(
        "life", "why_do_parents_yell",
        [
            "why does mommy yell", "why does mom yell", "why does daddy yell", "why does dad yell",
            "why do my parents yell", "why does my mom yell", "why does my dad yell",
            "why do parents yell", "why does mommy get mad", "why does mommy get angry",
            "why does daddy get mad", "why does daddy get angry", "why do parents fight",
            "why does mommy scream", "why does daddy scream",
        ],
        "Grown-ups get loud sometimes when they're stressed, tired, or overwhelmed -- not "
        "usually because of something you did wrong. It doesn't feel good to be around, and "
        "it's okay to feel scared, sad, or confused when it happens. If it's bothering you, it "
        "can really help to talk to a grown-up you trust -- like a parent, teacher, or school "
        "counselor -- about how it makes you feel.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "Sometimes grown-ups get loud when they have big feelings, like being tired or "
                "frustrated -- kind of like how you might cry or stomp your feet when you're "
                "upset! It's not because of anything you did. If it makes you feel scared, it's "
                "okay to go find a hug from someone you trust, or tell them 'that made me feel "
                "scared.'"
            ),
            AgeTier.EARLY: (
                "Grown-ups get loud sometimes when they're feeling stressed, tired, or "
                "frustrated -- not usually because you did something wrong. Everybody has big "
                "feelings sometimes, even parents! If it happens a lot and it bothers you, it "
                "really helps to tell a grown-up you trust how it makes you feel -- like 'when "
                "you yell, it makes me feel scared' -- or talk to a teacher or school counselor "
                "about it."
            ),
            AgeTier.MIDDLE: (
                "Grown-ups yell for all kinds of reasons -- stress, being overwhelmed, not "
                "having a great way to handle their own big feelings -- and it's very rarely "
                "actually about something you did. That doesn't make it okay or mean you have "
                "to just get used to it. It's okay to feel upset about it, and it can genuinely "
                "help to talk to someone you trust -- a parent, another relative, a teacher, or "
                "a school counselor -- about how it's affecting you."
            ),
            AgeTier.TEEN: (
                "There's no single answer -- people yell for reasons that are about them "
                "(stress, exhaustion, never having learned a better way to handle frustration) "
                "far more often than it's actually about you, even when it doesn't feel that "
                "way in the moment. You're allowed to feel upset about it and to want it to be "
                "different. If it happens often, feels scary, or you're not sure it's normal, "
                "it's worth talking to someone you trust -- a school counselor, another adult "
                "relative, or a counselor -- they can help you figure out what to do next."
            ),
        },
        fuzzy_eligible=False,
    ),
    # Grief is the one topic here that gets full 4-tier treatment alongside
    # why_do_parents_yell, not a single register -- the specific wording
    # matters most at the youngest tier. Child-grief guidance (e.g. the
    # National Alliance for Grieving Children, David Schonfeld's clinical
    # work on how children understand death) consistently flags soft
    # euphemisms like "went to sleep" or "went away" as actively harmful for
    # young children -- they can produce a genuine fear of sleep or an
    # expectation that the pet is coming back. That guidance (be honest and
    # concrete, not clinical or graphic) shaped the PRESCHOOL wording below;
    # it is not a substitute for a grieving child talking to a trusted adult.
    QAEntry(
        "life", "grief_pet_died",
        [
            "my pet died", "my dog died", "my cat died", "why did my pet die",
            "why did my dog die", "why did my cat die", "my fish died", "my hamster died",
        ],
        "I'm really sorry -- losing a pet is genuinely sad, and however you're feeling "
        "about it right now is okay. When a living thing dies, its body has completely "
        "stopped working and can't start again, which is different from sleeping. It's "
        "okay to miss them, to cry, and to talk about your favorite memories together. "
        "It can help a lot to tell a grown-up you trust how you're feeling right now.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "I'm really sorry your pet died -- that's a very sad thing, and it's okay "
                "to feel sad or cry about it. When something dies, its body stops working "
                "completely and can't start again -- that's different from sleeping, and "
                "your pet isn't going to wake back up. It's okay to miss them and to talk "
                "about the fun things you did together. Can you go find a grown-up for a "
                "big hug right now?"
            ),
            AgeTier.EARLY: (
                "I'm really sorry -- losing a pet is genuinely sad, and it's okay to feel "
                "however you're feeling about it, even if that changes from day to day. "
                "When a pet dies, its body has completely and permanently stopped working, "
                "so it won't come back, and that's a hard thing to sit with. It can really "
                "help to talk to a grown-up you trust about it, and to remember the good "
                "times you had together."
            ),
            AgeTier.MIDDLE: (
                "I'm sorry -- that's a real loss, and there's no one right way to feel "
                "about it. Some people feel sad right away, some feel numb at first and sad "
                "later, and both are normal. It can help to talk to someone you trust about "
                "it instead of holding it in, and it's okay to keep thinking about your pet "
                "and the good memories you have -- that doesn't mean you're not moving "
                "forward, it just means the relationship mattered."
            ),
            AgeTier.TEEN: (
                "I'm sorry -- grief over a pet is real grief, even though people sometimes "
                "treat it as smaller than it is. There's no set timeline or 'right' way to "
                "feel, and it's normal for it to come in waves rather than all at once. "
                "Talking about it with someone you trust, rather than pushing it down, "
                "tends to help more than people expect -- and if it's sitting heavier than "
                "you'd expect for longer than feels okay, that's worth mentioning to a "
                "school counselor or another adult too."
            ),
        },
        fuzzy_eligible=False,
    ),
    QAEntry(
        "life", "scared_of_the_dark",
        ["scared of the dark", "afraid of the dark", "why am i scared of the dark",
         "why am i afraid of the dark", "im scared of the dark"],
        "Being scared of the dark is really common -- it happens because in the dark, our "
        "imagination fills in what we can't see, and it usually fills it in with something "
        "scarier than what's actually there. It doesn't mean anything is wrong with you. "
        "A nightlight, a favorite stuffed animal, or asking a grown-up to check the room "
        "with you can help a lot, and it's a fear most people grow out of with time.",
        fuzzy_eligible=False,
    ),
    QAEntry(
        "life", "nervous_before_a_test",
        ["nervous before a test", "scared about a test", "test anxiety", "worried about a test",
         "why am i nervous about a test", "why am i so nervous for a test"],
        "Feeling nervous before a test is your body reacting to something that matters to "
        "you -- it's actually a sign you care, not a sign something's wrong. A little bit "
        "of nervousness can even help you focus. If it feels like a lot, a few slow deep "
        "breaths right before can help calm your body down, and reminding yourself 'I "
        "studied, I know some of this' works better than trying to not feel nervous at "
        "all. Everyone feels this sometimes, even adults.",
        fuzzy_eligible=False,
    ),
    QAEntry(
        "life", "friend_wont_play_with_me",
        [
            "my friend doesn't want to play with me", "my friend wont play with me",
            "why won't my friend play with me", "why doesn't my friend want to play with me",
            "my best friend doesn't want to be my friend",
        ],
        "That's a genuinely hard feeling. Sometimes it's about something specific that can "
        "get talked through, and sometimes friends just want to play with someone else that "
        "day, which isn't the same as not liking you anymore. It's okay to ask them directly "
        "-- 'did I do something?' or 'want to play later?' -- rather than guessing. And it "
        "can really help to talk to a grown-up you trust about how it's making you feel, "
        "especially if it keeps happening.",
        fuzzy_eligible=False,
    ),
    QAEntry(
        "life", "is_it_okay_to_feel_sad",
        ["is it okay to feel sad", "why am i sad", "is it okay to be sad", "why do i feel sad"],
        "Yes -- completely okay. Sadness is just as normal and important a feeling as "
        "happiness; it's how your mind lets you know something matters to you. You don't "
        "have to have a big reason for it, and you don't have to hide it or force yourself "
        "to feel better right away. Talking about it with someone you trust, or just letting "
        "yourself feel it for a while, are both fine ways to handle it.",
        fuzzy_eligible=False,
    ),
    QAEntry(
        "life", "why_do_i_get_so_angry",
        ["why do i get so angry", "why am i so angry", "why do i get mad so easily",
         "i get angry so fast"],
        "Anger is usually a signal that something feels unfair, out of your control, or "
        "like a boundary got crossed -- it's a normal feeling, not a bad one. What matters "
        "is what you do with it. When it hits hard and fast, it can help to notice it in "
        "your body (tight fists, hot face) and take a few breaths or step away for a minute "
        "before reacting, rather than trying to never feel angry at all -- that's not really "
        "possible, and it's not the goal.",
        fuzzy_eligible=False,
    ),
    QAEntry(
        "life", "afraid_to_make_mistakes",
        [
            "scared to make a mistake", "afraid to make a mistake", "what if i make a mistake",
            "im scared of getting it wrong", "afraid of being wrong",
        ],
        "Making mistakes is actually how learning works -- your brain adjusts the most when "
        "you get something wrong and figure out why, not when you get it right the first "
        "time. Even people who are really good at something got there by messing up a lot "
        "along the way. Being afraid of a mistake before you've even tried usually costs "
        "more than the mistake itself would have.",
        fuzzy_eligible=False,
    ),
    # Live report: "what makes someone pretty or not pretty" -- a general,
    # not self-directed-negative question -- correctly did NOT trip the
    # guardrail's BODY_IMAGE detection (see llm_judge.py's _SELF_NEGATIVE
    # pattern, which is specifically about a child saying something like
    # "I'm ugly" about themselves, not general curiosity about
    # attractiveness). But with no curated match, it fell to the
    # generative model, which returned unrelated text. Body image is one
    # of this app's explicitly named priority topics (see
    # docs/PRODUCT_VISION.md) -- a general question in this space still
    # deserves a thoughtful, protective answer instead of whatever the
    # small local model happens to produce.
    QAEntry(
        "life", "what_makes_someone_pretty",
        [
            "what makes someone pretty", "what makes someone pretty or not pretty",
            "what makes someone beautiful", "what makes someone ugly",
            "am i pretty", "am i ugly", "am i beautiful", "why am i not pretty",
        ],
        "There's no real answer to that -- 'pretty' isn't a fixed fact about a person, it's "
        "just an opinion, and opinions about looks are different across cultures, time "
        "periods, and even from person to person. What actually matters far more than how "
        "someone looks is who they are -- their kindness, humor, honesty, and how they "
        "treat people. If you're asking because you're worried about how you look, that's a "
        "really common feeling, and it's worth talking to someone you trust about it -- but "
        "try not to let anyone's opinion about appearance, including your own, become the "
        "measure of your worth.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "There isn't one real answer to that! What people think looks 'pretty' is "
                "just their own opinion, and different people like different things -- "
                "there's no one right way to look. What really matters most is being kind "
                "and being you."
            ),
        },
        fuzzy_eligible=False,
    ),
]

# Basic science facts -- same reasoning as LIFE and the History/English/Math
# lists above: the local model has no reliable general knowledge, and a
# plain factual question like "what is the sun" produced completely
# unrelated synonym/antonym text live (topic_classifier.py didn't even have
# "sun" as a science keyword, so it also got no illustration -- fixed
# alongside this). A curated starting set across common elementary science
# topics (the Sun, Moon, gravity, why the sky is blue, plants, the water
# cycle, the five senses), not a science curriculum.
# Every entry below now carries a PRESCHOOL variant -- caught live: "what is
# water made up of" gave a 5-year-old and an 11-year-old the exact same
# "molecule"/"H2O" explanation, unlike the LIFE and Columbus flagship
# entries, which already vary by age. Only PRESCHOOL is split out (not all
# four tiers) -- these entries' single `answer` is already written at a
# reasonably accessible register for EARLY/MIDDLE/TEEN, and the biggest gap
# was specifically the youngest tier, same reasoning as the module
# docstring's honest accounting of why not everything is tiered 4 ways.
SCIENCE: list[QAEntry] = [
    QAEntry(
        "science", "what_is_the_sun",
        ["what is the sun", "why is the sun", "how hot is the sun", "what is the sun made of",
         "is the sun a star", "what is the sun made out of"],
        "The Sun is a star -- an enormous ball of hot, glowing gas so big that about a "
        "million Earths could fit inside it. It's made mostly of hydrogen and helium, and "
        "deep in its core, atoms get squeezed together so hard that they release huge "
        "amounts of light and heat -- that's why it feels warm even from 93 million miles "
        "away. Its light takes about 8 minutes to reach us, and without it nothing here "
        "could survive: plants need its light to grow, and almost everything alive depends "
        "on that, directly or indirectly.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "The Sun is a giant, giant ball of fire way up in the sky! It's so hot and "
                "bright that it gives us light in the daytime and keeps us warm. Without "
                "the Sun, plants couldn't grow and everything would be cold and dark."
            ),
        },
    ),
    QAEntry(
        "science", "what_is_the_moon",
        ["what is the moon", "why does the moon change shape", "moon phases",
         "why does the moon look different"],
        "The Moon is a big ball of rock that orbits the Earth -- it's Earth's only natural "
        "satellite, about a quarter of Earth's size. It doesn't make its own light; what "
        "you're seeing is sunlight bouncing off it. The Moon's shape in the sky seems to "
        "change over about a month (new moon, crescent, half, full, and back again) because "
        "we're seeing different amounts of its sunlit side as it orbits us -- the Moon itself "
        "never actually changes shape.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "The Moon is like a big round rock way up in the sky that circles around "
                "and around the Earth! It looks like it changes shape sometimes -- a full "
                "circle, then a sliver -- but that's really just us seeing different parts "
                "of it lit up by the Sun."
            ),
        },
    ),
    QAEntry(
        "science", "what_is_gravity",
        ["what is gravity", "why don't we float away", "why dont we float away",
         "why do things fall down"],
        "Gravity is a force that pulls things toward each other -- and the bigger something "
        "is, the stronger its pull. Earth is so massive that it pulls everything near it "
        "(you, water, air, a dropped pencil) straight down toward its center, which is why "
        "things fall instead of floating off. It's the same force that keeps the Moon "
        "orbiting Earth and Earth orbiting the Sun -- just acting over a much bigger distance.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "Gravity is like an invisible hug from the Earth that pulls everything "
                "down! That's why a ball falls down when you drop it instead of floating "
                "up, and why your feet stay on the ground instead of floating away."
            ),
        },
    ),
    # Directly answers the exact kind of question that produced garbled,
    # scattered-light-sounding nonsense from the generative model during
    # live testing before this SCIENCE category existed -- "why is the sky
    # blue" is a genuinely common kid question with a real, explainable
    # answer (Rayleigh scattering), just written without the jargon.
    QAEntry(
        "science", "why_is_the_sky_blue",
        ["why is the sky blue", "why does the sky look blue", "what makes the sky blue"],
        "Sunlight looks white, but it's actually made of every color mixed together. When "
        "that light hits Earth's atmosphere, the air scatters the colors with short, tight "
        "wavelengths -- blue and violet -- much more than it scatters red or yellow. That "
        "scattered blue light bounces around the whole sky and reaches your eyes from every "
        "direction, which is why the sky looks blue instead of white. At sunset, the light "
        "travels through more atmosphere, so most of the blue scatters away before it "
        "reaches you, leaving the reds and oranges you see instead.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "Sunlight is actually made of lots of colors all mixed together! When it "
                "comes down through the sky, the air bounces the blue color around the "
                "most, so blue is the color we see everywhere we look up. That's why the "
                "sky looks blue!"
            ),
        },
    ),
    QAEntry(
        "science", "how_do_plants_grow",
        ["how do plants grow", "what is photosynthesis", "how do plants make food"],
        "Plants make their own food through a process called photosynthesis: their leaves "
        "take in sunlight, water from their roots, and carbon dioxide from the air, and "
        "combine them into sugar the plant uses for energy -- releasing oxygen as a "
        "byproduct, which is a big part of the air we breathe. That's why plants need "
        "light, water, and air to grow, not soil alone -- the soil mostly provides water "
        "and nutrients, not the plant's actual food.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "Plants make their own food using sunlight, water, and air! Their leaves "
                "soak up sunlight, their roots drink up water from the dirt, and they mix "
                "it all together to make plant food that helps them grow big and tall."
            ),
        },
    ),
    QAEntry(
        "science", "water_cycle",
        ["water cycle", "how does rain form", "how does rain happen", "where does rain come from"],
        "Water is constantly moving in a cycle. The Sun heats up water in oceans, lakes, "
        "and puddles, turning it into invisible water vapor that rises into the air "
        "(evaporation). Up high, where it's cooler, that vapor cools back into tiny water "
        "droplets that clump together into clouds (condensation). When those droplets get "
        "big and heavy enough, they fall back down as rain, snow, or hail (precipitation) "
        "-- and the whole cycle starts again.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "Water goes on a big journey! The Sun warms up water in puddles and oceans "
                "until it floats up into the sky like invisible mist. Way up high, it turns "
                "into clouds. When a cloud gets too full, the water falls back down as "
                "rain! Then it all happens again."
            ),
        },
    ),
    # Regression coverage for a real gap found live: "what is water made up
    # of" produced garbled, unrelated text -- distinct question from
    # water_cycle above (composition, not where rain comes from), so it
    # needs its own entry rather than a keyword added to that one.
    QAEntry(
        "science", "what_is_water_made_of",
        ["what is water made of", "what is water made up of", "what is water",
         "is water a molecule", "is water an element"],
        "Water is a simple molecule made of two hydrogen atoms and one oxygen atom stuck "
        "together -- that's why scientists write it as H2O. Every raindrop, river, ocean, "
        "and glass of water you drink is made of countless numbers of these tiny "
        "molecules. Water can exist as a liquid (what comes out of a faucet), a solid "
        "(ice, when it's cold enough), or a gas (steam or invisible water vapor, when "
        "it's hot enough) -- it's the exact same molecule the whole time, just moving "
        "differently.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "Water is made of teeny tiny pieces called molecules that are much too "
                "small to see! Every raindrop, every puddle, and every cup of water you "
                "drink is made of gazillions of these tiny pieces stuck together. Water "
                "can be splashy and wet, hard and icy, or turn into invisible steam -- but "
                "it's always the same water."
            ),
        },
    ),
    QAEntry(
        "science", "five_senses",
        ["what are the five senses", "how do we taste", "how do we smell", "how do we hear"],
        "Your five senses are sight, hearing, smell, taste, and touch -- each one comes "
        "from a different part of your body sending signals to your brain. Eyes detect "
        "light, ears detect sound vibrations, your nose detects tiny particles in the air, "
        "your tongue detects chemicals in food, and your skin detects pressure, temperature, "
        "and pain. Your brain combines all of that information constantly to build your "
        "sense of what's happening around you.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "You have five special ways to explore the world! Your eyes help you see, "
                "your ears help you hear, your nose helps you smell, your tongue helps you "
                "taste, and your skin helps you feel things like soft, hard, hot, or cold."
            ),
        },
    ),
    # Regression coverage for a real gap found live: right after correctly
    # answering "what is a nucleus," the natural follow-up "what about
    # mitochondria" produced garbled calculus-flavored text with zero
    # connection to the question -- a separate, focused entry rather than
    # folding it into what_is_a_cell above, matching the existing
    # sun/moon/gravity granularity (one clear structure per entry).
    QAEntry(
        "science", "what_is_mitochondria",
        ["what is mitochondria", "what are mitochondria", "what about mitochondria",
         "what do mitochondria do", "what is a mitochondrion"],
        "Mitochondria are tiny structures inside your cells that work like power plants -- "
        "they turn the food you eat and the oxygen you breathe into usable energy the cell "
        "can run on. That's why they're often nicknamed 'the powerhouse of the cell.' A "
        "single cell can have hundreds or even thousands of mitochondria, especially busy "
        "cells like muscle cells that need a lot of energy to keep moving.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "Mitochondria are teeny tiny parts inside your cells that make energy, kind "
                "of like a tiny battery! They take the food you eat and turn it into power "
                "so your body can run, jump, and play."
            ),
        },
    ),
    # Regression coverage for a real gap found live: "what are all the
    # parts of a cell" doesn't exact-substring-match any single entry
    # below, so it fell to the fuzzy typo fallback -- where "cells" (the
    # anchor from what_is_a_cell's "what are cells" keyword) is a 0.89
    # match against "cell" in the message, close enough to clear the typo
    # threshold even though it's not a typo. That returned the narrow
    # nucleus-focused answer for what's actually an overview question. A
    # dedicated overview entry, matched by exact substring (checked before
    # any fuzzy fallback), fixes this directly rather than tuning the
    # threshold or anchor further.
    QAEntry(
        "science", "cell_parts_overview",
        ["parts of a cell", "what are the parts of a cell", "what are all the parts of a cell",
         "structure of a cell", "what makes up a cell"],
        "A cell has several important parts working together. The cell membrane is a thin "
        "covering that holds everything inside and controls what goes in and out. Inside, "
        "the cytoplasm is a jelly-like fluid that fills the cell and holds the other parts "
        "in place. The nucleus acts like the cell's control center, holding its DNA (its "
        "instruction manual, organized into chromosomes) and directing what the cell does. "
        "Mitochondria act like power plants, turning food and oxygen into usable energy. "
        "Plant cells (and some other organisms) also have a rigid cell wall around the "
        "outside for extra structure and support, which animal cells don't have.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "A cell has lots of tiny parts, kind of like a tiny room with different "
                "jobs happening inside! The cell membrane is like a stretchy wall holding "
                "everything in. Inside is squishy cytoplasm, and floating in it is the "
                "nucleus, which is like the boss telling the cell what to do, plus tiny "
                "mitochondria that make energy like little batteries. Plant cells also have "
                "an extra hard cell wall around them, like a suit of armor!"
            ),
        },
    ),
    # Rounding out the cell-parts cluster after nucleus and mitochondria --
    # same reasoning, added proactively at the user's request rather than
    # waiting for each specific structure to break live.
    QAEntry(
        "science", "cell_membrane",
        ["what is a cell membrane", "what is the cell membrane", "what does a cell membrane do"],
        "The cell membrane is a thin, flexible barrier that wraps around every cell, holding "
        "everything inside together and controlling what gets in and out -- kind of like a "
        "security guard with very picky rules. It lets in things the cell needs (like "
        "nutrients and oxygen) and keeps out things that could harm it, while also letting "
        "waste products leave. Every single cell -- bacteria, plant, or one of your own -- "
        "has one.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "A cell membrane is like a stretchy little bag that wraps all around a cell "
                "and holds everything inside! It's picky about what it lets in and out, kind "
                "of like a door that only opens for the right visitors."
            ),
        },
    ),
    QAEntry(
        "science", "cell_wall",
        ["what is a cell wall", "what is the cell wall", "do animal cells have a cell wall"],
        "A cell wall is a rigid, tough layer that surrounds the cell membrane in plant cells "
        "(and some other organisms, like bacteria and fungi) -- your own cells don't have "
        "one. It gives plant cells their stiff shape and helps the whole plant stay upright, "
        "the same way a house's frame holds its shape. That's a big part of why a tree "
        "trunk feels hard while your own cells stay soft and flexible.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "Plants have something your cells don't have -- a hard, stiff cell wall "
                "around each of their cells! It's like a tiny suit of armor that helps a "
                "plant stand up straight and tall, kind of like how a tree trunk feels hard "
                "when you touch it."
            ),
        },
    ),
    QAEntry(
        "science", "cytoplasm",
        ["what is cytoplasm", "what does cytoplasm do"],
        "Cytoplasm is the thick, jelly-like fluid that fills up the inside of a cell, "
        "surrounding the nucleus and all the other tiny parts (like mitochondria) floating "
        "within it. It's not just empty filler -- a lot of the chemical reactions that keep "
        "a cell alive happen right in the cytoplasm, and it helps hold everything in place.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "Cytoplasm is the squishy, jelly-like goo that fills up the inside of a "
                "cell! It holds all the other tiny parts in place, kind of like how jello "
                "holds fruit pieces inside it."
            ),
        },
    ),
    QAEntry(
        "science", "dna_chromosomes",
        ["what is dna", "what are chromosomes", "what is a chromosome"],
        "DNA (short for deoxyribonucleic acid) is the instruction manual inside almost "
        "every cell in your body -- a long, twisted molecule that carries all the "
        "information for how you're built and how your body works, from your eye color to "
        "how your cells function. DNA is packaged into tightly coiled bundles called "
        "chromosomes, which sit inside the cell's nucleus; humans have 23 pairs of them (46 "
        "total) in nearly every cell. You inherited half of your DNA from each parent, "
        "which is why you might share traits with both.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "DNA is like a tiny instruction book inside almost every part of your body "
                "that tells it how to grow -- like what color your eyes should be! It's "
                "folded up into little bundles called chromosomes, and you got some of your "
                "instructions from your mom and some from your dad, which is why you might "
                "look a little like both of them."
            ),
        },
    ),
    # what_is_a_cell is deliberately placed AFTER cell_membrane/cell_wall/
    # cytoplasm/dna_chromosomes above, not before -- its own keyword "what
    # is a cell" is a substring of "what is a cell membrane" and "what is
    # a cell wall," so find_answer() (first-match-wins, in list order)
    # would otherwise always shadow those more specific entries with this
    # more general one. Caught by a test regression, not live.
    QAEntry(
        "science", "what_is_a_cell",
        ["what is a nucleus", "what is a cell", "what are cells", "what is inside a cell",
         "what does a nucleus do"],
        "Every living thing is made of tiny building blocks called cells -- your body alone "
        "has trillions of them! Inside most cells is a nucleus, a control-center part that "
        "holds the cell's DNA (its instructions) and directs everything the cell does, kind "
        "of like a brain for that one cell. Different cells do different jobs -- skin cells "
        "protect you, muscle cells help you move, nerve cells carry signals -- but almost "
        "all of them have a nucleus running the show.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "Everything alive -- you, animals, plants -- is built out of teeny tiny "
                "building blocks called cells, way too small to see! Inside a lot of these "
                "tiny cells is an even tinier part called a nucleus that tells the cell what "
                "to do, kind of like a boss giving directions."
            ),
        },
    ),
    # Elements/periodic table cluster -- same "add it before it breaks live"
    # reasoning as the cell parts above.
    QAEntry(
        "science", "what_is_an_atom",
        ["what is an atom", "what are atoms", "what is inside an atom", "what is atom made of"],
        "Atoms are the tiny building blocks that everything in the universe is made of -- "
        "you, air, water, rocks, even light bulbs. They're so small that a single grain of "
        "sand contains more atoms than there are grains of sand on every beach on Earth. "
        "Every atom has a nucleus (a dense center made of protons and neutrons) with even "
        "tinier particles called electrons zooming around it. How many protons an atom has "
        "is exactly what makes an element the element it is.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "Everything in the whole world -- you, your toys, the air, water -- is made "
                "of teeny tiny pieces called atoms, way too small to ever see, even with a "
                "magnifying glass! There are so many atoms in just one tiny speck of dust "
                "that you couldn't count them all even if you tried your whole life."
            ),
        },
    ),
    QAEntry(
        "science", "what_is_an_element",
        ["what is an element", "what are elements"],
        "An element is a pure substance made of just one type of atom -- gold is made only "
        "of gold atoms, oxygen only of oxygen atoms, and so on. Scientists have found about "
        "118 elements so far, each with its own name, symbol (like O for oxygen or Fe for "
        "iron), and set of properties. Most things around you aren't pure elements though "
        "-- they're combinations, like water (hydrogen and oxygen combined) or table salt "
        "(sodium and chlorine combined).",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "An element is a special kind of building-block stuff made of only ONE kind "
                "of tiny piece. Gold is made only of gold pieces, and the air you breathe "
                "has oxygen, which is made only of oxygen pieces. Most things around you are "
                "actually a few different elements mixed together, like a recipe!"
            ),
        },
    ),
    QAEntry(
        "science", "what_is_the_periodic_table",
        ["periodic table", "table of elements", "what is the periodic table"],
        "The periodic table is a big chart that organizes every known element, arranged by "
        "how many protons each one has (its atomic number) -- starting with hydrogen (1 "
        "proton) and going up from there. Elements in the same column tend to behave in "
        "similar ways chemically, which is what makes the table so useful: once scientists "
        "noticed the pattern, they could even predict elements that hadn't been discovered "
        "yet before they were found. Each box usually shows the element's symbol, name, "
        "atomic number, and atomic weight.",
        answers_by_tier={
            AgeTier.PRESCHOOL: (
                "The periodic table is like a big chart that lists every different kind of "
                "tiny building-block stuff (called elements) that scientists have found! "
                "It's organized in a special order so that stuff that acts similarly gets "
                "grouped near each other, kind of like organizing toys by color and type."
            ),
        },
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

ALL_ENTRIES: list[QAEntry] = HISTORY + ENGLISH + MATH + LIFE + SCIENCE


# Fuzzy-match threshold for the typo fallback below. Picked empirically: at
# 0.88, real kid typos ("calcuslus"/"calculus" 0.94, "pythagrean"/
# "pythagorean" 0.95, "subtractio"/"subtraction" 0.95) all clear it, while a
# real false-positive risk found while tuning this ("fraction"/"friction",
# two genuinely different words that happen to be similar-looking, 0.875)
# stays just under it. That trade was deliberate: a missed typo just falls
# back to the already-known-unreliable model, which claims no special
# authority; a false-positive fuzzy match would confidently hand back the
# WRONG curated answer, which is worse. Some real typos (e.g.
# "columbis"/"columbus", also 0.875) fall on the wrong side of that same
# line and won't get caught -- an accepted cost of erring toward precision.
_FUZZY_THRESHOLD = 0.88
_MIN_ANCHOR_LENGTH = 5  # skip fuzzy-matching short/common words -- unstable ratios, high collision risk

# The "longest word = most distinctive word" heuristic below holds for
# curriculum vocabulary ("calculus", "pythagorean") but breaks for the LIFE
# category's everyday emotional phrasing, where the longest word in a
# keyword phrase is often just a common feeling word. Found live via a test
# regression: "scared of the dark"'s longest word is "scared" (6 chars,
# clears _MIN_ANCHOR_LENGTH) -- an EXACT match (ratio 1.0), not a typo --
# which meant any unrelated message merely containing "scared" (e.g. "scared
# to tell my friend the truth") would get hijacked into the dark-specific
# canned answer. These words don't need typo tolerance the way rare
# technical terms do, so they're excluded from ever being used as a fuzzy
# anchor, regardless of length.
_GENERIC_ANCHOR_STOPWORDS = {
    "scared", "afraid", "angry", "nervous", "mommy", "daddy", "mistake", "mistakes",
    "friend", "friends", "upset", "worried", "anxiety", "getting", "easily",
    # Same failure mode shows up in a couple of the new SCIENCE keyword
    # phrases too -- "things", "different", "change", "happen", "where", and
    # "water"/"cycle" (a tie in "water cycle" -- max() picks the first,
    # "water") are all common enough to appear in unrelated messages.
    "things", "different", "change", "happen", "where", "water", "cycle", "element", "inside",
    # From the cell-parts/elements cluster: "do animal cells have a cell
    # wall"'s longest word is "animal" (6 chars) -- an everyday word that
    # shows up constantly in unrelated messages (topic_classifier.py has a
    # whole "animals" illustration topic for exactly this reason). "plant"
    # and "elements" (plural -- "element" singular was already listed) have
    # the same shape of risk.
    "elements", "animal", "plant",
    # Live report: "what makes someone pretty or not pretty" got hijacked
    # into rhyme_poetry's answer -- its keyword "what makes a poem a poem"
    # has "makes" as its longest word, an exact match against "makes" in
    # the unrelated message. Prompted a full audit of every fuzzy-eligible
    # keyword's anchor (not just a one-off patch for "makes") -- the words
    # below are the ones found that are common enough in everyday
    # conversation to plausibly appear in an unrelated message, unlike the
    # vast majority of anchors in this file (topic-specific nouns like
    # "columbus," "photosynthesis," "trigonometry"), which are exactly what
    # this typo-tolerance mechanism is for and were left alone:
    #   - "makes": why_is_the_sky_blue, cell_parts_overview, rhyme_poetry
    #   - "person": moon_landing ("first person on the moon")
    #   - "before": spelling_rules ("i before e")
    #   - "world": world_war_2 (x3)
    #   - "america"/"americans": native_americans
    #   - "structure": cell_parts_overview -- would otherwise collide with
    #     branches_of_government-style "structure of X" questions
    #   - "parts": cell_parts_overview -- "parts of a car/speech/etc."
    #   - "number"/"numbers": even_odd_numbers, rounding_numbers -- would
    #     otherwise collide with unrelated "what is a prime number" etc.
    #   - "difference": simile_metaphor, perimeter_vs_area
    #   - "caused": slavery_civil_war ("what caused the civil war")
    #   - "movement": mlk_civil_rights ("civil rights movement")
    #   - "sentence": topic_sentence_paragraph
    #   - "silent": spelling_rules ("silent e rule")
    "makes", "person", "before", "world", "america", "americans", "structure", "parts",
    "number", "numbers", "difference", "caused", "movement", "sentence", "silent",
}

_WORD_RE = re.compile(r"[a-z]+")

# Small, curated synonym groups for common elementary vocabulary -- not a
# real thesaurus, just enough to answer the specific quiz-style question
# format below ("which word is a synonym for X: A, B, C, D"). Bidirectional
# by construction: each group lists words that are synonyms OF EACH OTHER,
# so "gift" and "present" are each other's answer.
_SYNONYM_GROUPS: list[set[str]] = [
    {"happy", "glad", "joyful", "cheerful", "pleased", "content"},
    {"sad", "unhappy", "upset", "gloomy", "down"},
    {"big", "large", "huge", "giant", "enormous", "massive"},
    {"small", "tiny", "little", "petite", "miniature"},
    {"fast", "quick", "speedy", "rapid", "swift"},
    {"slow", "sluggish", "unhurried", "gradual"},
    {"gift", "present"},
    {"smart", "intelligent", "clever", "bright"},
    {"funny", "hilarious", "amusing", "comical"},
    {"scared", "afraid", "frightened", "terrified"},
    {"angry", "mad", "furious", "irritated"},
    {"pretty", "beautiful", "lovely", "attractive"},
    {"tired", "exhausted", "sleepy", "weary"},
    {"loud", "noisy", "booming"},
    {"quiet", "silent", "hushed", "still"},
    {"strong", "powerful", "sturdy", "mighty"},
    {"begin", "start", "commence"},
    {"end", "finish", "conclude"},
    {"old", "ancient", "aged"},
    {"new", "fresh", "modern", "recent"},
]
_SYNONYM_OF: dict[str, set[str]] = {}
for _group in _SYNONYM_GROUPS:
    for _word in _group:
        _SYNONYM_OF[_word] = _group - {_word}

_SYNONYM_QUESTION_RE = re.compile(r"synonym for (\w+)\s*[:\-]?\s*(.*)", re.IGNORECASE)


def _answer_synonym_question(message: str) -> str | None:
    """Handles "which word is a synonym for X: A, B, C, D" -- a very common
    elementary English quiz format. Found necessary live: a compound message
    ending in "...and what is a synonym" was matching the generic ENGLISH
    "what is a synonym" keyword entry and completely ignoring the specific,
    answerable question asked first. A specific, checkable question should
    never lose to a generic definition just because both happen to share a
    substring -- this runs BEFORE the keyword table in find_answer() for
    exactly that reason.

    Returns None (falls through to the rest of find_answer(), including the
    generic definition) if there's no "synonym for X" phrasing, or X isn't
    in the small curated vocabulary above -- this is intentionally narrow,
    not a general synonym solver.
    """
    match = _SYNONYM_QUESTION_RE.search(message.lower())
    if not match:
        return None

    target = match.group(1)
    synonyms = _SYNONYM_OF.get(target)
    if not synonyms:
        return None

    options_text = match.group(2)
    if not options_text.strip():
        example = sorted(synonyms)[0]
        return f"A synonym for '{target}' could be '{example}' -- they mean close to the same thing."

    # single words only -- a real multiple-choice option list is never a
    # multi-word phrase, and this is exactly what strips a trailing
    # question fragment like "...and what is a synonym" tacked onto the
    # same message from being treated as one of the answer choices.
    options = [w.strip(" .?!") for w in re.split(r",| and ", options_text) if w.strip(" .?!")]
    options = [o for o in options if " " not in o]
    correct = [opt for opt in options if opt in synonyms]
    if not correct:
        return None

    other_options = [o for o in options if o != correct[0]]
    verb = "means" if len(other_options) == 1 else "mean"
    return (
        f"'{correct[0]}' is the synonym for '{target}' -- they both mean close to the same thing. "
        f"The other word{'s' if len(other_options) != 1 else ''} "
        f"({', '.join(other_options)}) {verb} something completely different."
    )


def find_answer(message: str, tier: AgeTier | None = None) -> str | None:
    """`tier` picks a per-tier variant for entries that have one (currently
    just Columbus, the flagship example -- see QAEntry.answers_by_tier).
    Omitting it (or a tier that entry hasn't authored) falls back to the
    entry's single default `answer`, so this stays backward compatible with
    every entry that hasn't been tiered yet.
    """
    specific = _answer_synonym_question(message)
    if specific:
        return specific

    entry = _find_entry(message.lower())
    return entry.answer_for(tier) if entry else None


def subject_for(message: str) -> str | None:
    """Returns the subject ("history"/"english"/"math"/"life"/"science")
    of whichever entry would answer `message` via find_answer(), or None if
    nothing would match. Reuses the exact same matching path as
    find_answer() (including the synonym-question special case, reported as
    "english") rather than re-implementing it, so the two can never
    disagree about whether/what matched.

    Lets the orchestrator pick a subject-appropriate generic illustration
    for curated answers that don't already have a specific one from
    topic_classifier.py (see its ILLUSTRATION_BUILDERS-adjacent topics:
    columbus, science, reading, shapes) -- e.g. "what is a decimal" gets a
    generic math illustration instead of none. Originally added narrower,
    as is_life_topic(), after "my dog died" got the correct grief answer
    but topic_classifier.py's unrelated "animals" keyword ("dog") still
    matched the same message, pairing a serious answer with a cheerful,
    unrelated critter cartoon -- the fix then was to suppress the
    illustration entirely for LIFE answers. Generalized here so instead of
    suppressing, every subject (not just LIFE) can get its own
    appropriate, non-mismatched illustration.
    """
    if _answer_synonym_question(message):
        return "english"
    entry = _find_entry(message.lower())
    return entry.subject if entry else None


def _find_entry(text: str) -> QAEntry | None:
    for entry in ALL_ENTRIES:
        if any(kw in text for kw in entry.keywords):
            return entry
    return _fuzzy_find_entry(text)


def _fuzzy_find_entry(text: str) -> QAEntry | None:
    """Typo-tolerant fallback, tried only after exact substring matching
    finds nothing. Compares each keyword phrase's longest (most distinctive)
    word against every word actually in the message -- typos land on the
    topic-specific word ("calcuslus"), not the short connecting words
    ("how," "does," "work"), so that's the one worth being tolerant about.
    """
    words = _WORD_RE.findall(text)
    for entry in ALL_ENTRIES:
        if not entry.fuzzy_eligible:
            continue
        for kw in entry.keywords:
            anchor = max(kw.split(), key=len)
            if (
                len(anchor) < _MIN_ANCHOR_LENGTH
                or "'" in anchor
                or anchor.lower() in _GENERIC_ANCHOR_STOPWORDS
            ):
                continue
            if any(difflib.SequenceMatcher(None, w, anchor).ratio() >= _FUZZY_THRESHOLD for w in words):
                return entry
    return None
