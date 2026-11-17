package com.hexanima.npu;

import android.os.Bundle;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

/** One place for all the "why is it like this" notes that used to clutter the main screen. */
public class Info extends Base {
    static final String[][] ITEMS = {
        {"Шаги и Turbo LoRA", "Steps and the Turbo LoRA",
         "Загружаются только базовые модели. Ускорение даёт отдельная Turbo LoRA: выберите её файл (.safetensors) в строке под моделью, подойдёт любая LoRA для этой модели — например Lightning для Qwen, 8-step DMD для MOSS-Image, Turbo v1.1 для Anima. Число шагов задаётся вручную в блоке «Шаги»: поставьте то, на что рассчитана LoRA (обычно 4 или 8). Без Turbo LoRA нужно больше шагов (у Anima ещё работает CFG). Меньше шагов, чем нужно LoRA, даёт мутную картинку, больше — только замедляет. Силу Turbo LoRA можно менять ползунком.",
         "Only base models are downloaded. Speed comes from a separate Turbo LoRA: pick its file (.safetensors) in the row under the model; any LoRA made for that model works, for example Lightning for Qwen, 8-step DMD for MOSS-Image, Turbo v1.1 for Anima. The number of steps is set by hand in the Steps block: use what the LoRA was made for (usually 4 or 8). Without a Turbo LoRA more steps are needed (Anima also uses CFG). Fewer steps than the LoRA expects look blurry, more only cost time. The Turbo LoRA strength has a slider."},
        {"Какой размер можно", "Which sizes work",
         "Ширина и высота кратны 16, число патчей (ширина/16 × высота/16) делится на 32 и не больше 6144. Примеры: 512×768, 640×640, 1024×1024, 960×1536. Кнопка ⇄ меняет портрет на альбом. Большие размеры считаются дольше и декодируются плитками.",
         "Width and height are multiples of 16, the patch count (width/16 × height/16) is divisible by 32 and at most 6144. Examples: 512×768, 640×640, 1024×1024, 960×1536. The ⇄ button swaps portrait and landscape. Larger sizes take longer and are decoded in tiles."},
        {"Seed", "Seed",
         "-1 — каждый раз случайный. Число повторяет ту же картинку при том же описании, размере и модели. Использованный seed виден под картинкой и в истории.",
         "-1 is random every time. A number repeats the same picture with the same prompt, size and model. The seed used is shown under the picture and in the history."},
        {"Почему первые шаги быстрее", "Why the first steps are faster",
         "Первые два шага считаются в уменьшенном разрешении (общая композиция), последние два — в полном, поэтому детали сохраняются. На некоторых нестандартных размерах это отключается само.",
         "The first two steps run at reduced resolution (overall composition), the last two at full resolution, so details are kept. On some unusual sizes this switches off automatically."},
        {"Загрузить и выгрузить модель", "Loading and unloading",
         "Загруженная модель занимает несколько гигабайт оперативной памяти. Если телефон тормозит или вы закончили — нажмите «Выгрузить». При генерации модель загрузится сама (~5 с).",
         "A loaded model holds several gigabytes of RAM. If the phone slows down or you are done, tap Unload. It loads itself when you generate (~5 s)."},
        {"Где хранятся картинки", "Where pictures are stored",
         "Только внутри приложения. В галерею телефона (Pictures/Anima) картинка попадает после «Скачать». Удалённое уходит в корзину и хранится 30 дней.",
         "Only inside the app. A picture goes to the phone gallery (Pictures/Anima) after Download. Deleted pictures go to the trash and stay there for 30 days."},
        {"Негативный промпт", "Negative prompt",
         "Поле негативного промпта есть у базовых моделей Anima без Turbo LoRA: там работает CFG. С Turbo LoRA CFG выключен (scale 1), негативный промпт ничего не меняет, поэтому поле скрыто. Нежелательное лучше просто не писать в описании.",
         "The negative prompt field is shown for base Anima models without a Turbo LoRA, where CFG is used. With a Turbo LoRA CFG is off (scale 1), a negative prompt would change nothing and the field is hidden. Simply leave unwanted things out of the prompt."},
        {"Теги качества и подсказки", "Quality tags and suggestions",
         "Перед описанием автоматически добавляются теги качества (их можно изменить в настройках). В поле описания при вводе появляются подсказки тегов Danbooru, а кнопка «Шаблоны» хранит сохранённые и недавние описания.",
         "Quality tags are added before your prompt automatically (change them in Settings). While you type, Danbooru tag suggestions appear, and the Templates button keeps saved and recent prompts."},
        {"Теги качества: только Anima", "Quality tags: Anima only",
         "Теги качества из настроек добавляются только к описаниям для моделей Anima. MOSS-Image их не получает, а пока она выбрана, этой настройки в списке нет.",
         "The quality tags from Settings are added only to prompts for Anima models. MOSS-Image does not get them, and the setting is hidden while MOSS-Image is selected."},
        {"MOSS-Image", "MOSS-Image",
         "MOSS-Image 5B (Apache-2.0) — вторая, более крупная модель: описывайте картинку обычными предложениями, а не тегами. Текстовый энкодер, трансформер и декодер работают на NPU. Нужно около 14 ГБ на накопителе; проверено на телефоне с 12 ГБ ОЗУ. Веса читаются с накопителя при каждой картинке: около 30 с для 512×768 и около 80 с для 1 Мпикс. Загружать модель заранее не нужно. Для 4 или 8 шагов выберите файл Turbo LoRA (rapid-moss — 4 шага, 8-step DMD от lodestones — 8 шагов) и поставьте то же число шагов.",
         "MOSS-Image 5B (Apache-2.0) is a second, larger model: describe pictures in plain sentences, not tags. The text encoder, the transformer and the decoder all run on the NPU. It needs about 14 GB of storage and was tested on a phone with 12 GB of RAM. Its weights are streamed from storage for every picture: about 30 s for 512×768, about 80 s for 1 Mpx. There is nothing to preload. For 4 or 8 steps pick a Turbo LoRA file (rapid-moss: 4 steps, the 8-step DMD by lodestones: 8 steps) and set the same number of steps."},
        {"Размеры MOSS-Image", "MOSS-Image sizes",
         "Подходит любой размер: обе стороны кратны 16, не меньше 256, не больше 6144 патчей (ширина/16 × высота/16), например 1536×1024. Turbo-адаптеры обучали примерно на 1 Мпикс (1024×1024, 832×1216…), но на меньших размерах картинки тоже получаются. Размеры без готового декодера декодируются плитками, поэтому тон может слегка сместиться.",
         "Any size works: both sides are multiples of 16, at least 256, at most 6144 patches (width/16 × height/16), for example 1536×1024. The turbo adapters were trained at about 1 Mpx (1024×1024, 832×1216…), but smaller sizes also work. Sizes without a ready decoder are decoded in tiles, so the tone can shift slightly."},
        {"LoRA для Anima", "LoRA for Anima",
         "Выберите свой файл .safetensors, сделанный для Anima (форматы kohya, diffusers/PEFT, LoKr и LoHa; 28 или 40 блоков). Файл проверяется сразу после выбора. Сила 1,00 — как задумал автор. Можно выбрать Turbo LoRA и обычную LoRA вместе: они сворачиваются в веса за один проход. Свёртка идёт при загрузке: первая картинка с новым набором (или новой силой) ждёт около 12–13 с, статус «Подготавливаю LoRA…». Готовые веса хранятся 8-битными блоками (около 2 ГБ на диске) только для последнего набора: возврат к прежнему набору снова займёт эти секунды. Часть LoRA для текстового энкодера не применяется.",
         "Pick your own .safetensors file made for Anima (kohya, diffusers/PEFT, LoKr and LoHa formats; 28 or 40 blocks). It is checked as soon as you pick it. Strength 1.00 is the author's setting. A Turbo LoRA and a regular LoRA can be used together: they are folded into the weights in one pass. Folding happens on load: the first picture with a new set (or a new strength) waits about 12-13 s, with the status \"Preparing the LoRA…\". The folded weights are kept as 8-bit blocks (about 2 GB of storage) for the last set only: going back to an earlier set takes those seconds again. The text-encoder part of a LoRA is not applied."},
        {"LoRA для MOSS-Image", "LoRA for MOSS-Image",
         "Выберите свой файл .safetensors, сделанный именно для MOSS-Image (LoRA для FLUX, SDXL и других моделей не подходят). Файл проверяется сразу после выбора, а если он не подходит, приложение объяснит почему. Сила 1,00 — как задумал автор: меньше — слабее эффект, больше — сильнее. Первая картинка с новой LoRA или новой силой тратит около 15 с на подготовку (временные файлы около 300 МБ). В слотах помещается LoRA ранга до 128; одна LoRA заменяет Turbo-слот. Использование LoRA регулируется лицензией её автора.",
         "Pick your own .safetensors file made for MOSS-Image (LoRAs for FLUX, SDXL and other models do not fit). It is checked as soon as you pick it, and if it does not fit the app says why. Strength 1.00 is the author's setting: lower is weaker, higher is stronger. The first picture with a new LoRA or strength spends about 15 s preparing it (temporary files of about 300 MB). The slots take a LoRA of rank up to 128; one LoRA replaces the Turbo slot. Using a LoRA is subject to its author's license."},
        {"Qwen-Image-2.1-Turbo", "Qwen-Image-2.1-Turbo",
         "Qwen-Image 2.1 Turbo (команда Qwen) — модель, которая хорошо рисует надписи и понимает длинные описания обычными предложениями. Для 4 шагов выберите файл Turbo LoRA (например Lightning от LightX2V) в строке под моделью; свёртка занимает секунды и хранится в маленьком кэше (около 16 МБ). Текст, ViT, DiT и VAE работают на NPU; нужно около 14 ГБ памяти, рекомендуется телефон с 12 ГБ ОЗУ. Только некоммерческое использование (Qwen Research License).",
         "Qwen-Image 2.1 Turbo (Qwen team) draws legible text in pictures and follows long prompts written as plain sentences. For 4 steps pick a Turbo LoRA file (for example Lightning by LightX2V) in the row under the model; preparing it takes seconds and is kept in a small cache (about 16 MB). The text encoder, the ViT, the transformer and the VAE all run on the NPU; it needs about 14 GB of storage and a phone with 12 GB of RAM is recommended. Non-commercial use only (Qwen Research License)."},
        {"Редактирование фото", "Editing photos",
         "Выберите фото в блоке «Редактирование фото» и напишите, что изменить: «сделай небо закатным», «убери человека справа». Картинка собирается из вашего фото и описания; размер результата подбирается по форме фото, а сам снимок уменьшается примерно до 0,4 Мпикс. Фото, из которого не получилось нужное, лучше пробовать с другим seed. Без фото модель рисует картинку с нуля. Редактирование идёт примерно в 1,3 раза дольше обычной генерации.",
         "Pick a photo in the Edit a photo block and write what to change: \"make the sky a sunset\", \"remove the person on the right\". The result is built from your photo and the prompt; the output size follows the photo's shape and the photo itself is reduced to about 0.4 Mpx. If the first try is not what you wanted, try another seed. Without a photo the model draws from scratch. Editing takes about 1.3 times as long as plain generation."},
        {"Размеры Qwen-Image", "Qwen-Image sizes",
         "Обе стороны кратны 16, число патчей (ширина/16 × высота/16) делится на 32 и не больше 4608, например 1024×1024 или 768×1152. При редактировании фото и результат вместе не должны превышать примерно 6800 патчей.",
         "Both sides are multiples of 16, the patch count (width/16 × height/16) is divisible by 32 and at most 4608, for example 1024×1024 or 768×1152. When editing, the photo and the result together must stay within about 6800 patches."},
        {"Mage-Flow-Edit", "Mage-Flow-Edit",
         "Mage-Flow-Edit-Turbo (Microsoft, 4B, лицензия MIT) рисует картинки и правит фото по описанию за 4 шага, около 12 секунд на картинку. Текст, ViT, DiT и VAE работают на NPU; загрузка около 8,7 ГБ. Чтобы изменить фото, выберите его в блоке «Редактирование фото» и опишите, что поменять; можно добавить ещё до двух картинок-референсов и сослаться на них в описании («картинка 1», «картинка 2»).",
         "Mage-Flow-Edit-Turbo (Microsoft, 4B, MIT license) draws pictures and edits photos by a prompt in 4 steps, about 12 seconds a picture. The text encoder, the ViT, the transformer and the VAE all run on the NPU; the download is about 8.7 GB. To edit a photo, pick it in the Edit a photo block and describe what to change; you can add up to two more reference pictures and refer to them in the prompt (\"picture 1\", \"picture 2\")."},
        {"Размеры Mage-Flow", "Mage-Flow sizes",
         "Обе стороны кратны 16, число патчей (ширина/16 × высота/16) делится на 32. Модель считает окнами, поэтому жёсткого предела размера нет: у картинки и у правки вместе с фото (каждое фото занимает столько же патчей, сколько результат) может быть до 24576 патчей в сумме, например 2048×2048 (около 3,5 минут) или 1024×768 с тремя фото. Чем больше размер, тем дольше: время растёт примерно как квадрат числа патчей.",
         "Both sides are multiples of 16 and the patch count (width/16 × height/16) is divisible by 32. The model works in windows, so there is no hard limit on the size: a picture, or an edit together with its pictures (each picture takes as many patches as the result), may have up to 24576 patches in all, for example 2048×2048 (about 3.5 minutes) or 1024×768 with three pictures. The bigger the size, the longer: the time grows about with the square of the patch count."},
        {"LTX-2.5 Video", "LTX-2.5 Video",
         "LTX-2.5 (Lightricks, 22B, дистиллированная) делает короткое видео по описанию: 17–97 кадров (по умолчанию 17), частота кадров в MP4 на выбор, без звука. Текстовый энкодер Gemma, 48 блоков DiT (8 шагов, без CFG) и VAE работают на NPU; весь клип занимает около полутора минут. Модель есть в каталоге загрузок (около 31 ГБ; лицензия LTX-2 Community License). Результат: MP4 рядом с кадром-превью в истории (кнопка ▶).",
         "LTX-2.5 (Lightricks, 22B, distilled) makes a short video from a prompt: 17–97 frames (17 by default), a frame rate of your choice for the MP4, no sound. The Gemma text encoder, the 48 DiT blocks (8 steps, no CFG) and the VAE run on the NPU; a whole clip takes about a minute and a half. The model is in the download catalog (about 31 GB; LTX-2 Community License). The result is an MP4 next to a preview frame in the history (the ▶ button)."},
        {"MiniMax H3 Ref2VA Video", "MiniMax H3 Ref2VA Video",
         "MiniMax-H3 Ref2VA (33B) делает видео по описанию и, по желанию, по фото-референсу: человек, животное или предмет с фото появляется в видео, а окружение и движение берутся из описания. 22–90 кадров, 4 шага (дистилляционная turbo-LoRA, без CFG), без звука. Скачивается на экране «Модели» после MiniMax-H3-Image: текстовый энкодер у них общий. Лицензия MiniMax H3 Community: не для США, ЕС, Великобритании и Кореи.",
         "MiniMax-H3 Ref2VA (33B) makes a video from a prompt and, optionally, a reference photo: the person, animal or object in the photo appears in the video while the setting and the motion come from the prompt. 22–90 frames, 4 steps (distilled turbo LoRA, no CFG), no sound. It is downloaded from the Models screen after MiniMax-H3-Image: they share the text encoder. MiniMax H3 Community licence: not for the US, EU, UK and Korea."},
        {"Размеры Ref2VA", "Ref2VA sizes",
         "Обе стороны кратны 32, минимум 256; токенов в кадре (ширина/32 × высота/32) должно быть кратно 32, например 512×512, 768×512, 1024×576, 768×768, 1024×1024. Всего токенов (кадры × токены кадра) не больше 16000: 1024×576 даёт 56 кадров за ~20 минут, 512×512 — 22 кадра за ~5 минут. Длина клипа и размер меняют время почти квадратично.",
         "Both sides are multiples of 32, at least 256; the tokens per frame (width/32 × height/32) must be divisible by 32, e.g. 512×512, 768×512, 1024×576, 768×768, 1024×1024. At most 16000 tokens in total (latent frames × tokens per frame): 1024×576 gives 56 frames in ~20 minutes, 512×512 gives 22 frames in ~5 minutes. Clip length and size change the time almost quadratically."},
        {"Размеры LTX", "LTX sizes",
         "Обе стороны кратны 32, минимум 256; число токенов 3 × (ширина/32) × (высота/32) делится на 32, например 512×384, 384×512, 512×512, 768×512, 768×768. Описание лучше писать подробно и по-английски: кто, что делает, свет, камера.",
         "Both sides are multiples of 32, at least 256; the token count 3 × (width/32) × (height/32) must be divisible by 32, for example 512×384, 384×512, 512×512, 768×512, 768×768. Write the prompt in detail, in English: who, what they do, the light, the camera."},
        {"FLUX.3 Image Klein", "FLUX.3 Image Klein",
         "FLUX.3 Image Klein (Black Forest Labs, 5,6B) — фотореалистичная модель: понимает длинные описания обычными предложениями и рисует надписи. 16 шагов без CFG: 512×512 примерно за минуту, 1024×1024 за 3–4 минуты. Текст, трансформер и декодер работают на NPU. Модель есть в каталоге загрузок (около 9 ГБ). Лицензия FLUX Kommunity: только некоммерческое использование.",
         "FLUX.3 Image Klein (Black Forest Labs, 5.6B) is a photorealistic model that follows long prompts written as plain sentences and draws text. 16 steps, no CFG: about a minute for 512×512, 3–4 minutes for 1024×1024. The text encoder, the transformer and the decoder run on the NPU. The model is in the download catalog (about 9 GB). FLUX Kommunity licence: non-commercial use only."},
        {"Размеры Klein", "Klein sizes",
         "Обе стороны кратны 16, число патчей (ширина/16 × высота/16) делится на 32, не больше 6144 (например 1536×1024), минимум 256 по каждой стороне. Описание лучше писать подробно, одним связным абзацем.",
         "Both sides are multiples of 16, the patch count (width/16 × height/16) is divisible by 32 and at most 6144 (for example 1536×1024), at least 256 on each side. Write the prompt in detail, as one connected paragraph."},
        {"Какие телефоны подходят", "Supported phones",
         "Проверено на Snapdragon 8 Gen 1 / 8+ Gen 1 (Hexagon V69). Root не нужен. Остальные чипы не поддерживаются.",
         "Tested on Snapdragon 8 Gen 1 / 8+ Gen 1 (Hexagon V69). No root needed. Other chips are not supported."},
        {"Больше кадров в видео (Увеличение → Видео)", "More frames in a video (Upscale → Video)",
         "В меню «⋮» → «Увеличение» → «Видео» нейросеть RIFE v4.6 (hzwer, MIT) дорисовывает кадры между каждой парой кадров вашего ролика: ×2 или ×4, движение становится плавнее (12 → 24 или 48 кадров в секунду). Работает на NPU, звук сохраняется, результат попадает в галерею. Подходят видео до 1280×768 (720p); для каждого размера скачивается своя маленькая сеть (12–40 МБ). Время: около полутора минут на 3 секунды HD-видео при ×4, в несколько раз быстрее для малых размеров. Жёсткие склейки кадров не интерполируются. Видео без изменения размера: резкость не улучшается, только плавность.",
         "In the menu → Upscale → Video the RIFE v4.6 network (hzwer, MIT) draws new frames between every pair of frames of your clip: x2 or x4, so motion gets smoother (12 → 24 or 48 frames per second). It runs on the NPU, the sound is kept and the result goes to the gallery. Videos up to 1280x768 (720p) work; a small network (12-40 MB) is downloaded for each size. About a minute and a half for 3 seconds of HD video at x4, several times faster for small sizes. Hard cuts are not interpolated. The picture size does not change: only the smoothness improves."},
        {"Теги из картинки (кнопка с фото)", "Tags from a picture (photo button)",
         "Кнопка с картинкой над полем описания выбирает фото и определяет его теги Danbooru моделью WD EVA02 Tagger v3 (SmilingWolf, Apache-2.0): они подставляются в описание (для Anima), заменяя его или добавляясь к нему. Модель работает на NPU, около 3 секунд, скачивается один раз (674 МБ). Порог: обычные теги от 0,35, персонажи от 0,85. Оценок (rating) в тегах нет.",
         "The picture button above the prompt field picks a photo and finds its Danbooru tags with the WD EVA02 Tagger v3 model (SmilingWolf, Apache-2.0): they replace the prompt or are added to it (for Anima). The model runs on the NPU, about 3 seconds, and is downloaded once (674 MB). Thresholds: normal tags from 0.35, characters from 0.85. Ratings are not included."},
        {"Лицензии", "Licenses",
         "Модели MiaoMiao (автор MIAOKA, Civitai) построены на Anima от CircleStone Labs и NVIDIA Cosmos. Только некоммерческое использование. Автор модели может потребовать убрать её из каталога.",
         "MiaoMiao models (by MIAOKA, Civitai) are built on Anima by CircleStone Labs and NVIDIA Cosmos. Non-commercial use only. A model author can ask for removal from the catalog."},
        {"Лицензии MOSS-Image", "MOSS-Image licenses",
         "MOSS-Image (OpenMOSS), Qwen3.5-9B (команда Qwen), адаптер rapid-moss (darask0, lodestones), малый декодер FLUX.2 (Black Forest Labs) и квантизации Unsloth — все под Apache License 2.0. Файлы — конвертированные для NPU копии; изменения перечислены в уведомлениях ниже. Тексты лицензий внизу, в «Лицензиях компонентов».",
         "MOSS-Image (OpenMOSS), Qwen3.5-9B (Qwen team), the rapid-moss adapter (darask0, lodestones), the FLUX.2 small decoder (Black Forest Labs) and the Unsloth quantisations are all under the Apache License 2.0. The files are converted copies made for the NPU; the changes are listed in the notices below. License texts are at the bottom, under Component licenses."},
    };

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        LinearLayout col = screen(L.s("Инфо", "Info"), null);
        about(col);
        for (String[] it : ITEMS) {
            LinearLayout c = card(col);
            c.addView(text(L.s(it[0], it[1]), 16, TXT, true));
            TextView body = text(L.s(it[2], it[3]), 14, DIM, false); body.setPadding(0, dp(6), 0, 0); body.setLineSpacing(0, 1.15f);
            c.addView(body);
        }
    }

    void about(LinearLayout col) {
        LinearLayout c = card(col);
        String ver = "";
        try { ver = getPackageManager().getPackageInfo(getPackageName(), 0).versionName; } catch (Exception ignored) { }
        c.addView(text(L.s("О приложении", "About"), 16, TXT, true));
        TextView v = text("HexAnima " + ver + "\n" + L.f("Чип %s · Android %s", "Chip %s · Android %s", Compat.soc(), android.os.Build.VERSION.RELEASE), 13, DIM, false); v.setPadding(0, dp(6), 0, dp(8)); c.addView(v);
        Button log = button(L.s("Отправить лог", "Send the log"), FIELD, ACC); log.setTextSize(13);
        log.setOnClickListener(x -> sendLog());
        c.addView(log, new LinearLayout.LayoutParams(-1, dp(42)));
        TextView lh = text(L.s("Авторы и ссылки", "Authors and links"), 14, TXT, true); lh.setPadding(0, dp(14), 0, dp(2)); c.addView(lh);
        String[][] links = {
            {"MiaoMiao 3D Harem (MIAOKA)", "https://civitai.red/models/431957/miaomiao-3d-harem"},
            {"MiaoMiao Harem (MIAOKA)", "https://civitai.red/models/934764/miaomiao-harem"},
            {"MiaoMiao RealSkin (MIAOKA)", "https://civitai.red/models/2026594/miaomiao-realskin"},
            {"Anima (CircleStone Labs)", "https://huggingface.co/circlestone-labs/Anima"},
            {"MOSS-Image-SFT (OpenMOSS)", "https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT"},
            {"MOSS-Image 4-step adapter (darask0)", "https://huggingface.co/darask0/MOSS-Image_4-step_adapter_lora"},
            {"MOSS-Image 8-step DMD (lodestones)", "https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd"},
            {"Qwen3.5-9B (Qwen team)", "https://huggingface.co/Qwen/Qwen3.5-9B"},
            {"Unsloth GGUF quantisations", "https://huggingface.co/unsloth/MOSS-Image-SFT-GGUF"},
            {"FLUX.2 small decoder (Black Forest Labs)", "https://huggingface.co/black-forest-labs/FLUX.2-small-decoder"},
        };
        for (String[] l : links) {
            TextView t = text(l[0], 13, ACC, false); t.setPadding(0, dp(8), 0, 0);
            t.setOnClickListener(x -> startActivity(new android.content.Intent(android.content.Intent.ACTION_VIEW, android.net.Uri.parse(l[1]))));
            c.addView(t);
        }
        TextView hh = text(L.s("Лицензии компонентов", "Component licenses"), 14, TXT, true); hh.setPadding(0, dp(14), 0, dp(2)); c.addView(hh);
        String[][] lic = {
            {"Anima — CircleStone Labs Non-Commercial License; NVIDIA Open Model License (Cosmos)", ""},
            {"MiaoMiao models — by MIAOKA, non-commercial use, see the model pages", ""},
            {"Turbo LoRAs are chosen by the user as separate files (for example Anima Turbo v1.1 by somedoby, TURBO for ANIMA V4-29B by n_Arno, Lightning by LightX2V) and stay under their authors' licenses. The Anima-based models are modified versions of Anima (CircleStone Labs Non-Commercial License), not an official CircleStone Labs product", ""},
            {"Qwen3 0.6B text encoder, Qwen-Image VAE — Apache 2.0", ""},
            {"MOSS-Image-SFT (OpenMOSS), rapid-moss adapter (darask0) and its 8-step DMD base (lodestones), Qwen3.5-9B (Qwen team), Unsloth GGUF quantisations, FLUX.2 small decoder (Black Forest Labs) — all Apache License 2.0. The MOSS-Image pack holds converted copies (re-packed for the NPU, the adapter kept unmerged); user LoRAs stay under their authors' licenses", ""},
            {"Qwen-Image-2.1-Turbo, its VAE and the Lightning v2-RL adapter (LightX2V) — Qwen RESEARCH LICENSE AGREEMENT, non-commercial use only; Qwen3-VL-8B-Instruct and its vision tower, Unsloth GGUF quantisations — Apache License 2.0. The Qwen-Image pack holds converted copies (re-packed for the NPU, the adapter kept unmerged)", ""},
            {"Mage-Flow-Edit-Turbo and Mage-VAE (Microsoft) — MIT License; its Qwen3-VL-4B text encoder and vision tower (Qwen team) — Apache License 2.0. The Mage-Flow pack holds converted copies (re-packed for the NPU)", ""},
            {"LTX-2.5 (Lightricks) — LTX-2 Community License Agreement (free for organisations under 10 million USD annual revenue); its Gemma 4 12B text encoder (Google) — Gemma terms. The files are converted copies (INT8 ConvRot packs re-tiled for the NPU)", ""},
            {"FLUX.3 Image Klein (Black Forest Labs) — FLUX Kommunity License, non-commercial use only; its Qwen3-VL-4B text encoder (Qwen team) — Apache License 2.0", ""},
            {"MiniMax-H3 Ref2VA (MiniMax) — MiniMax-H3 Community License Agreement (the converted video model is downloaded from the Models screen; its Qwen3-VL text encoder comes from the MiniMax-H3-Image pack)", ""},
            {"Real-ESRGAN (Xintao Wang et al.), the upscaler networks — BSD 3-Clause License", ""},
            {"SeedVR2-3B (ByteDance Seed), the diffusion upscaler converted for the NPU — Apache License 2.0", ""},
            {"RIFE v4.6 (hzwer, Practical-RIFE), the frame interpolation network converted for the NPU — MIT License", ""},
            {"WD EVA02-Large Tagger v3 (SmilingWolf), the picture-to-tags network converted for the NPU — Apache License 2.0", ""},
            {"stable-diffusion.cpp — MIT", ""},
            {"TAEHV preview decoder — see github.com/madebyollin/taehv", ""},
            {"Qualcomm AI Engine Direct (QNN) runtime libraries — Qualcomm AI Stack License (redistributed only inside this app)", ""},
            {"Tag suggestions — Danbooru tag list from DominikDoom/a1111-sd-webui-tagcomplete", ""},
        };
        for (String[] l : lic) { TextView t = text("•  " + l[0], 12, DIM, false); t.setPadding(0, dp(6), 0, 0); c.addView(t); }
        for (String[] q : new String[][]{{"Qualcomm AI Stack License", "licenses/qualcomm_ai_stack_license.txt"}, {L.s("Уведомления QNN (сторонние компоненты)", "QNN notices (third-party components)"), "licenses/qnn_notice.txt"}, {"Apache License 2.0", "licenses/apache-2.0.txt"}, {"Qwen RESEARCH LICENSE AGREEMENT", "licenses/qwen_research_license.txt"}, {L.s("Уведомления Qwen-Image (изменения)", "Qwen-Image notices (changes made)"), "licenses/qwen_notice.txt"}, {L.s("Уведомления MOSS-Image (изменения)", "MOSS-Image notices (changes made)"), "licenses/moss_notice.txt"}, {L.s("Уведомления Mage-Flow (изменения)", "Mage-Flow notices (changes made)"), "licenses/mage_notice.txt"}, {"FLUX Kommunity License", "licenses/flux_kommunity_license.txt"}, {L.s("Уведомления FLUX.3 Klein (изменения)", "FLUX.3 Klein notices (changes made)"), "licenses/klein_notice.txt"}, {"LTX-2 Community License Agreement", "licenses/ltx2_community_license.txt"}, {L.s("Уведомления LTX-2.5 (изменения)", "LTX-2.5 notices (changes made)"), "licenses/ltx_notice.txt"}, {"Real-ESRGAN — BSD 3-Clause License", "licenses/real_esrgan_license.txt"}, {"RIFE — MIT License", "licenses/rife_license.txt"}}) {
            TextView t = text(q[0], 13, ACC, false); t.setPadding(0, dp(8), 0, 0);
            t.setOnClickListener(x -> showAsset(q[0], q[1]));
            c.addView(t);
        }
    }

    void showAsset(String title, String path) {
        String body;
        try (java.io.InputStream in = getAssets().open(path)) { body = new String(in.readAllBytes(), "UTF-8"); } catch (Exception e) { body = "(none)"; }
        android.widget.ScrollView sv = new android.widget.ScrollView(this);
        TextView t = text(body, 11, DIM, false); t.setPadding(dp(18), dp(14), dp(18), dp(14)); t.setTextIsSelectable(true);
        sv.addView(t);
        builder().setTitle(title).setView(sv).setPositiveButton(L.s("Закрыть", "Close"), null).show();
    }

    /** share the device info and the tails of the engine logs as plain text */
    void sendLog() {
        StringBuilder sb = new StringBuilder();
        try { sb.append("HexAnima ").append(getPackageManager().getPackageInfo(getPackageName(), 0).versionName); } catch (Exception ignored) { }
        sb.append("\nChip ").append(Compat.soc()).append(", Android ").append(android.os.Build.VERSION.RELEASE).append(", ").append(android.os.Build.MODEL)
          .append(String.format(", RAM %.1f GB, free %.1f GB\n", Compat.ramGb(this), Compat.freeGb(this)));
        sb.append("Installed: ").append(new Models(getFilesDir()).installed()).append("\n");
        sb.append("App RAM ").append(String.format("%.1f GB", Compat.appRamGb())).append("\n");
        for (String n : new String[]{"animad.log", "sdcli.log", "gen.log"}) sb.append("\n==== ").append(n).append(" ====\n").append(tail(new java.io.File(getFilesDir(), "anima/" + n), 12000));
        sb.append("\n==== moss.log ====\n").append(tail(new java.io.File(Moss.dir(getFilesDir()), "moss.log"), 8000));
        sb.append("\n==== qwen.log ====\n").append(tail(new java.io.File(Qwen.dir(getFilesDir()), "qwen.log"), 8000));
        sb.append("\n==== mage.log ====\n").append(tail(new java.io.File(Mage.dir(getFilesDir()), "mage.log"), 8000));
        sb.append("\n==== ltx.log ====\n").append(tail(new java.io.File(Ltx.dir(getFilesDir()), "ltx.log"), 8000));
        sb.append("\n==== r2v.log ====\n").append(tail(new java.io.File(R2v.dir(getFilesDir()), "r2v.log"), 8000));
        sb.append("\n==== klein.log ====\n").append(tail(new java.io.File(Klein.dir(getFilesDir()), "klein.log"), 8000));
        android.content.Intent s = new android.content.Intent(android.content.Intent.ACTION_SEND).setType("text/plain").putExtra(android.content.Intent.EXTRA_SUBJECT, "HexAnima log").putExtra(android.content.Intent.EXTRA_TEXT, sb.toString());
        startActivity(android.content.Intent.createChooser(s, L.s("Отправить лог", "Send the log")));
    }

    static String tail(java.io.File f, int max) {
        try (java.io.RandomAccessFile r = new java.io.RandomAccessFile(f, "r")) {
            long from = Math.max(0, r.length() - max); r.seek(from);
            byte[] b = new byte[(int) (r.length() - from)]; r.readFully(b);
            return new String(b, "UTF-8");
        } catch (Exception e) { return "(none)"; }
    }
}
