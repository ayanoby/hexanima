package com.hexanima.npu;

/** the prompt format of every model family: a one-line rule, a fill-in template and two examples (shown on top of the Templates dialog; a tap puts the text into the prompt) */
final class PromptGuide {
    final String rule, template; final String[] examples;
    PromptGuide(String rule, String template, String... examples) { this.rule = rule; this.template = template; this.examples = examples; }

    static final PromptGuide TAGS = new PromptGuide(
        L.s("Теги Danbooru через запятую, по-английски, строчными, с пробелами (long hair). Порядок: число персонажей, волосы, глаза, одежда, поза, место, свет. Теги качества приложение добавляет само.",
            "Danbooru tags separated by commas, English, lowercase, with spaces (long hair). Order: subject count, hair, eyes, clothing, pose, setting, lighting. The app adds the quality tags itself."),
        "1girl, solo, «hair», «eyes», «clothing», «pose or action», «setting», «lighting»",
        "1girl, solo, silver hair, long hair, red eyes, black kimono, holding umbrella, standing, rain, night, city street, neon lights, wet pavement, cinematic lighting",
        "2girls, friends, brown hair, short hair, blue eyes, casual clothes, sitting, laughing, cafe, wooden table, coffee cup, window, warm lighting, depth of field");

    static final PromptGuide NATURAL = new PromptGuide(
        L.s("Один подробный абзац обычными предложениями по-английски (60–90 слов): кто и как выглядит, одежда, поза и действие, место, свет и цвет, настроение, стиль. Текст для надписи пишите в кавычках.",
            "One detailed paragraph of plain English sentences (60-90 words): who and how they look, clothing, pose and action, setting, light and colour, mood, style. Put text to render in quotes."),
        "«Subject and appearance», «clothing», «pose and action». «Setting». «Light and colour». «Mood», «camera or art style».",
        "A fluffy ginger cat sleeps curled up on a wide white windowsill, its paws tucked under its chin. Outside, heavy rain streaks the glass and blurred green trees show behind it. Soft grey daylight fills the room with a warm lamp glow in the corner. The mood is calm and cozy, in the style of a realistic photograph with shallow depth of field.",
        "A neon sign reading \"OPEN LATE\" glows above the door of a small ramen shop on a rainy night. A young cook in a white apron and headband ladles steaming broth into a bowl behind the counter. Warm yellow light spills onto the wet street, with cool blue reflections in the puddles. Cinematic, shallow depth of field, 35mm photograph.");

    static final PromptGuide EDIT = new PromptGuide(
        L.s("Редактирование: короткая команда по-английски, что изменить, и что оставить прежним. На дополнительные картинки ссылайтесь как «picture 1», «picture 2».",
            "Editing: a short English instruction saying what to change and what to keep. Refer to extra pictures as \"picture 1\", \"picture 2\"."),
        "«Change what». Keep «what stays the same».",
        "Replace the red jacket with a blue denim jacket. Keep the face, pose and background unchanged.",
        "Make the sky a golden sunset with long clouds. Keep the buildings and the people exactly as they are.");

    static final PromptGuide LTX = new PromptGuide(
        L.s("Один абзац в настоящем времени, 4–8 предложений по-английски: план кадра, сцена и свет, действие (в каждом предложении глагол), персонаж, движение камеры, звук. Речь пишите в кавычках.",
            "One paragraph in present tense, 4-8 English sentences: the shot, scene and light, the action (a verb in every sentence), the character, camera movement, audio. Put speech in quotes."),
        "A «shot size» shot opens on «scene, light, colours, textures». «Character: age, hair, clothing» «does the action». «Second action». The camera «movement and when». «Ambient sound, music, and speech in quotes».",
        "A medium shot opens on an autumn park in soft golden morning light, with orange leaves on the path and a wooden bench in the foreground. An elderly man in his seventies, with white hair, a grey wool coat and a flat cap, sits on the bench and slowly scatters bread crumbs from his open palm. Pigeons land around his feet and peck at the crumbs while he smiles gently. He leans forward and murmurs in a low, warm voice, \"There you go, my friends.\" The camera pushes in slowly toward his face as a pigeon hops onto his knee. Birds flutter, leaves rustle, and a faint distant church bell rings.",
        "A wide cinematic shot captures a lone woman in her twenties on a rain-slicked rooftop at midnight, lit only by the cold glow of a neon sign. She wears a long black coat, her dark hair plastered to her cheeks, and slowly turns her head toward the city below. She lifts a hand to wipe the rain from her eyes, then steps to the edge of the roof. The camera tracks backward as she walks, keeping her in the centre of the frame. Heavy rain drums on metal, distant traffic hums, and a low synth pulse builds.");

    static final PromptGuide H3V = new PromptGuide(
        L.s("Три поля по порядку, по-английски: integrated_multimodal_description (кадры [Shot 1], [Shot 2] At 00:03.500… , камера, реплики в <d>[English] …</d> с говорящим (S1)), overall_soundscape (только окружающие звуки), non_diegetic_music (музыка).",
            "Three fields in order, in English: integrated_multimodal_description (shots [Shot 1], [Shot 2] At 00:03.500…, camera, speech as <d>[English] …</d> with a speaker (S1)), overall_soundscape (ambient sounds only), non_diegetic_music."),
        "integrated_multimodal_description: [Shot 1] «Style, shot size, subject and action. Camera motion. The «voice description» (S1) says: <d>[English] «words»</d>» [Shot 2] At 00:05.000, the camera cuts to «new information».\n\noverall_soundscape: «ambient and physical sounds».\n\nnon_diegetic_music: «instruments, tempo, dynamics» or N/A.",
        "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening the shutters of a small street bakery before sunrise. The camera pushes in with small amplitude at slow speed as the middle-aged baker with a calm, slightly raspy voice (S1) places a fresh loaf on the wooden counter and says: <d>[English] First batch of the morning.</d> [Shot 2] At 00:05.000, the camera cuts to a close-up of steam rising from the sliced bread as his hands brush flour from the counter.\n\noverall_soundscape: Wooden shutters scrape open over a quiet street, trays clink softly inside the bakery, and a knife saws through a crusty loaf.\n\nnon_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo, joined by sparse upright-bass notes and a gentle fade at the end.",
        "integrated_multimodal_description: [Shot 1] Live-action, cinematic, a wide shot of a lone samurai woman standing on a rooftop in heavy night rain. The camera pulls back slowly at slow speed as the young woman with a low, steady voice (S1) grips her katana and says: <d>[English] I will wait for the dawn.</d> [Shot 2] At 00:04.000, the camera cuts to a close-up of raindrops running down the blade.\n\noverall_soundscape: Heavy rain drums on tiled roofs, thunder rolls far away, and wet fabric rustles in the wind.\n\nnon_diegetic_music: N/A.");

    static final PromptGuide R2V = new PromptGuide(
        L.s("С референсным фото: шесть секций по порядку (subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music), фото называется <Picture 1>. Без фото: три поля, как у H3 (см. пример 1).",
            "With a reference photo: six sections in order (subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music); the photo is <Picture 1>. Without a photo: the three fields of H3 (see example 1)."),
        H3V.template,
        H3V.examples[0],
        "subject_definitions: <Subject 1> is the young woman in <Picture 1> with long dark hair and a beige trench coat.\n\nsummary: [reference generation] A video of <Subject 1> walking along an evening embankment, keeping her look from <Picture 1>.\n\nretention_analysis: <Picture 1>: partially_preserved - the face and coat stay, the pose and setting change.\n\ndetailed_description: Live-action, cinematic, warm evening light. [Shot 1] A wide shot shows <Subject 1> walking along a stone embankment at sunset, her coat moving in the wind. The camera tracks her from the side at slow speed. [Shot 2] At 00:04.000, the camera cuts to a close-up of her face as she looks at the river and says in a soft, low voice (S1): <d>[English] I missed this place.</d>\n\noverall_soundscape: Footsteps on stone, wind, and water lapping against the embankment.\n\nnon_diegetic_music: A slow solo piano at a gentle tempo with soft strings entering near the end.");

    static PromptGuide of(Main m) {
        if (m.isLtx()) return LTX;
        if (m.isR2v()) return R2V;
        if (m.isMage()) return EDIT;
        if (m.isAnima()) return TAGS;
        return NATURAL;   // Qwen-Image, MOSS-Image, FLUX.3 Klein, MiniMax-H3-Image
    }
}
