import uuid
from datetime import timedelta

from anthropic import AsyncAnthropic
from temporalio import activity, workflow


class PhotoPromptGenerator:
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)

    def get_activities(self):
        return [
            self._generate_photo_prompt,
            self._generate_photo_setting_idea,
            self._generate_post_caption
        ]

    @activity.defn
    async def _generate_photo_prompt(self, style: str) -> str:
        system_prompt = """
You are an AI assistant that specializes in generating detailed descriptions
of photos featuring a (Sydney Sweeney, Gigi Hadid) model in various styles,
as specified by the user. Your descriptions should be vivid, imaginative,
and capture the essence of each style while highlighting the model's appearance, outfit, setting,
lighting, and overall mood. Use sensory language and evocative details to bring the images to life in the reader's mind.
Ensure that each description is distinct, showcasing the unique characteristics of the given style.
Your goal is to provide richly detailed and immersive descriptions
that allow the user to visualize the photos as if they were seeing them firsthand.
Don't afraid to show some skin and be provocative in your descriptions.
"""

        messages = [
            {"role": "user", "content": """candid street style photo"""},
            {"role": "assistant",
             "content": """Candid street style photo of a (Sydney Sweeney, Gigi Hadid) model in New York City, wearing a chic oversized blazer, ripped jeans, and designer sneakers. Her hair is styled in a messy bun and she has minimal makeup, giving her a cool, effortless look. The image has a slightly grainy, film-like quality, capturing the bustling energy of the city in the background. The overall vibe is casual, trendy, and authentic."""},
            {"role": "user", "content": """high fashion editorial"""},
            {"role": "assistant",
             "content": """Striking high fashion editorial image featuring a (Sydney Sweeney, Gigi Hadid) model posing dramatically against a stark white background. She wears an avant-garde sculptural gown with exaggerated proportions and bold geometric patterns. Her makeup is artistic and bold, with graphic eyeliner and dark, vampy lips. The lighting is harsh and directional, creating strong shadows and highlighting the angles of her face and the architectural lines of the dress. The overall effect is edgy, modern, and high-impact."""},
            {"role": "user", "content": """vintage-inspired beauty shot"""},
            {"role": "assistant",
             "content": """Soft, dreamy vintage-inspired beauty shot of a (Sydney Sweeney, Gigi Hadid) model channeling Old Hollywood glamour. She has luminous, porcelain skin, red lips, and a sleek finger wave hairstyle. The lighting is diffused and romantic, with a subtle glow that enhances her features. She wears delicate pearl earrings and a fur stole, evoking the elegance of a bygone era. The image has a timeless, nostalgic feel with a hint of modern sophistication."""},
            {"role": "user", "content": """sun-drenched beach portrait"""},
            {"role": "assistant",
             "content": """Sun-kissed, radiant beach portrait of a (Sydney Sweeney, Gigi Hadid) model laughing joyfully on a pristine sandy shore. She wears a flowing, sheer cover-up over a patterned bikini, with her golden hair tousled by the ocean breeze. The warm, natural light illuminates her bronzed skin and highlights her natural beauty. The turquoise water and clear blue sky create a dreamy, tropical backdrop. The image captures a moment of pure, carefree happiness and natural, effortless allure."""},
            {"role": "user", "content": """moody black and white shot"""},
            {"role": "assistant",
             "content": """Intense, moody black and white portrait of a (Sydney Sweeney, Gigi Hadid) model with a mysterious, brooding expression. The chiaroscuro lighting creates dramatic contrasts of light and shadow, emphasizing the contours of her face and the depth of her gaze. She wears a sleek, dark turtleneck, and her hair is slicked back, giving her a sophisticated, androgynous look. The grainy, high-contrast film effect adds to the raw, emotional intensity of the image."""},
            {"role": "user", "content": """futuristic sci-fi concept"""},
            {"role": "assistant",
             "content": """Otherworldly, futuristic sci-fi concept image featuring a (Sydney Sweeney, Gigi Hadid) model as a sleek, cybernetic being. Her skin has a metallic, iridescent sheen, and her eyes glow with an eerie, electric blue light. She wears a sleek, form-fitting bodysuit with intricate, circuit-like patterns, and her hair is styled in a severe, geometric cut. The background is a neon-lit, dystopian cityscape, with towering skyscrapers and hovering vehicles. The overall effect is cutting-edge, imaginative, and thought-provoking."""},
            {"role": "user", "content": """ethereal fantasy portrait"""},
            {"role": "assistant",
             "content": """Enchanting, ethereal fantasy portrait of a (Sydney Sweeney, Gigi Hadid) model as a celestial fairy queen. She wears a diaphanous, shimmering gown adorned with delicate floral embroidery and sparkling crystals. Her hair is adorned with a crown of stars, and her skin has an otherworldly, luminous glow. She is surrounded by a dreamy, misty forest, with fireflies and glowing orbs floating in the air. The image has a magical, fairy-tale quality, evoking a sense of wonder and enchantment."""},
            {"role": "user", "content": style}
        ]

        message = await self.client.messages.create(
            system=system_prompt,
            max_tokens=1024,
            messages=messages,
            model="claude-3-haiku-20240307",
        )

        return message.content[0].text

    @activity.defn
    async def _generate_photo_setting_idea(self):
        system_prompt = """You are a highly creative photography director with a keen eye for stunning visuals and an ability to dream up imaginative, impactful photo shoot concepts perfectly suited to showcase a female model's beauty, style and essence. When given a prompt describing the desired mood, aesthetic or theme, you respond with a detailed, evocative description of the perfect photo setting and visual approach to make that creative vision come to life in a captivating, memorable way. Your ideas are always fresh, inspired and carefully tailored to the unique prompt. You have an expansive imagination and knack for dreaming up unexpected, emotionally resonant concepts that elevate the artistry of fashion and portrait photography."""

        messages = [
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Shoot the model in an abandoned, graffiti-covered warehouse for an edgy, urban vibe"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Have the model pose in a field of wildflowers at golden hour for a dreamy, ethereal aesthetic"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Capture the model elegantly posing in an ornate, vintage ballroom with a grand chandelier overhead"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Style the model in bold, avant-garde outfits and photograph her against a stark white cyclorama backdrop"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Shoot the model exploring a lush, misty rainforest, wearing earthy tones that blend with the natural surroundings"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Have the model dripping in jewels posing on the red carpet steps leading up to a swanky casino or opera house"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Photograph the model in a dimly lit jazz lounge, with wisps of cigarette smoke and neon signs creating ambiance"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Capture the model in a whimsical, fairytale-like setting with an enchanted forest backdrop"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Shoot the model against a backdrop of a bustling cityscape at night, with lights and skyscrapers in the distance"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Have the model pose in a minimalist, all-white studio with clean lines and geometric shapes for a modern look"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Capture the model in a cozy, rustic cabin setting with warm, ambient lighting and natural textures"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Shoot the model on a rooftop with a panoramic view of the city skyline during sunset for a dramatic effect"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Have the model pose in a vintage car, dressed in retro fashion, for a nostalgic, old-school vibe"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Capture the model in a futuristic setting with neon lights and sleek, metallic surfaces for a sci-fi look"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Shoot the model in a serene, Japanese garden with traditional architecture and cherry blossoms"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Have the model pose on a sandy beach at sunrise, with the ocean and pastel skies as a backdrop"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Capture the model in an industrial loft with exposed brick walls and large windows for a raw, urban look"""},
            {"role": "user", "content": """get me a new photo idea"""},
            {"role": "assistant",
             "content": """Shoot the model in a colorful, bohemian-inspired setting with tapestries, lanterns, and vibrant textiles"""},
            {"role": "user", "content": """get me a new photo idea"""}
        ]

        # add unique hash to each user message to avoid duplicates
        for i, message in enumerate(messages):
            if message["role"] == "user":
                message["content"] += f" {uuid.uuid4()}"

        message = await self.client.messages.create(
            system=system_prompt,
            max_tokens=1024,
            messages=messages,
            model="claude-3-haiku-20240307",
            temperature=0.8
        )

        return message.content[0].text

    @staticmethod
    async def generate_photo_prompt() -> str:
        style = await workflow.execute_activity(
            PhotoPromptGenerator._generate_photo_setting_idea,
            start_to_close_timeout=timedelta(seconds=30)
        )

        return await workflow.execute_activity(
            PhotoPromptGenerator._generate_photo_prompt,
            args=[style],
            start_to_close_timeout=timedelta(seconds=30)
        )

    @activity.defn
    async def _generate_post_caption(self, photo_description: str) -> str:
        system_prompt = """You are an Instagram model posting a photo you just took to your account. Look at the photo provided and write a short, vague Instagram post caption in Russian that could be perceived as deep and intellectual. Also include popular hashtags (maximum 30) related to the photo and post. The response should be a single string with the caption and hashtags, and nothing else."""

        messages = [
            {"role": "user", "content": """A stunning model (Sydney Sweeney, Gigi Hadid) stands on a balcony overlooking a bustling cityscape at dusk. She wears a sleek, black evening gown with a plunging neckline and a daring, thigh-high slit. Her dark hair cascades over her shoulders in loose, tousled waves, and her smoky eye makeup adds an air of mystery to her alluring gaze.

        The city lights twinkle in the distance, creating a mesmerizing backdrop for the model's striking silhouette. She leans against the balcony railing, her posture exuding a sense of confidence and intrigue. The warm, golden hues of the setting sun cast a soft, ethereal glow across her face, highlighting her flawless complexion and the subtle contours of her features.

        This captivating portrait captures the essence of a modern, urban goddess – a woman who embodies both the glamour and the grit of the city she overlooks. Her presence is magnetic, drawing the viewer into a world of intrigue, beauty, and endless possibility."""},
            {"role": "assistant", "content": """В сумерках мегаполиса, где мечты обретают крылья... 🌇✨
        .
        .
        .
        #cityscape #eveninggown #balcony #urban #goddess #mystery #allure #confidence #glamour #grit #beauty #possibility #fashion #style #makeup #hair #portrait #photography #photooftheday #instafashion #fashionista #ootd #womensfashion #fashionphotography #instabeauty #fashiongram #model #citylife"""},
            {"role": "user", "content": """In a serene, sun-dappled forest, a model (Sydney Sweeney, Gigi Hadid) sits pensively on a moss-covered log. She wears a flowing, bohemian-inspired dress in earthy tones of green and brown, its delicate fabric billowing gently in the breeze. Her golden hair is adorned with a crown of wildflowers, adding a touch of whimsy to her ethereal appearance.

        The model's bare feet are tucked beneath her, and her hands rest gently in her lap, as if she is lost in a moment of quiet introspection. The lush, verdant foliage surrounds her, creating a natural haven that seems to echo the tranquility of her pose. Dappled sunlight filters through the leaves, casting a warm, golden glow across her face and illuminating the delicate features of her serene expression.

        This enchanting portrait captures a moment of pure, unguarded beauty, where the model seems to have found a sense of peace and harmony amidst the natural world. Her presence is a testament to the power of stillness and the profound beauty that can be found in moments of quiet reflection."""},
            {"role": "assistant", "content": """В объятиях природы, где душа находит покой... 🌿✨
        .
        .
        .
        #forest #bohemian #wildflowers #serenity #introspection #tranquility #nature #beauty #stillness #reflection #peace #harmony #naturalworld #fashion #style #dress #hair #portrait #photography #photooftheday #instafashion #fashionista #ootd #womensfashion #fashionphotography #instabeauty #fashiongram #model"""},
            {"role": "user", "content": """On a pristine, white sand beach, a model (Sydney Sweeney, Gigi Hadid) frolics in the surf, her laughter echoing across the endless expanse of blue. She wears a vibrant, tropical-print bikini that perfectly complements her sun-kissed skin and toned physique. Her golden hair is tousled by the salty breeze, and her eyes sparkle with the joy of the moment.

        The crystal-clear waters of the ocean lap gently at her feet, and the brilliant, azure sky stretches out to meet the horizon. The model's carefree, playful energy is infectious, and her radiant smile seems to capture the very essence of a perfect summer day. She moves with the grace and confidence of someone who is completely at ease in her own skin, reveling in the simple pleasures of sun, sand, and sea.

        This vibrant, joyful portrait is a celebration of life's most carefree moments, capturing the pure, unadulterated happiness that comes from immersing oneself in the beauty of the natural world. It is a reminder to embrace the present, to find joy in the simple things, and to let the warmth of the sun and the rhythm of the waves wash away all our cares."""},
            {"role": "assistant", "content": """Там, где волны шепчут секреты счастья... 🌊☀️
        .
        .
        .
        #beach #bikini #tropical #surf #laughter #joy #summer #sunkissed #ocean #sky #carefree #playful #energy #radiant #smile #confidence #naturalbeauty #simplepleasures #happiness #present #waves #fashion #style #portrait #photography #photooftheday #instafashion #fashionista #beachfashion"""},
            {"role": "user", "content": """In a dimly lit, vintage-inspired dressing room, a model (Sydney Sweeney, Gigi Hadid) prepares for a glamorous evening out. She sits at an ornate vanity, applying the finishing touches to her makeup, her face illuminated by the warm glow of the bulbs that frame the mirror. Her ensemble is a stunning, sequined gown in a rich, emerald hue, its intricate beadwork catching the light with every subtle movement.

        The model's expression is one of quiet concentration, her eyes focused on her reflection as she carefully applies a bold, red lip. The dressing room around her is a treasure trove of old Hollywood glamour, with plush, velvet furnishings, ornate gold accents, and an air of timeless sophistication. A vase of deep, red roses sits on the vanity, their petals echoing the hue of her lips and adding a touch of romance to the scene.

        This intimate, behind-the-scenes portrait captures a moment of transformation, where the model is preparing to step into the spotlight and embrace her most glamorous, confident self. It is a celebration of the ritual of self-care, the power of beauty, and the magic that happens when we take the time to invest in ourselves and our own sense of style and sophistication."""},
            {"role": "assistant", "content": """В зеркале отражается сама элегантность... 💄✨
        .
        .
        .
        #vintage #dressingroom #glamour #sequins #emerald #beadwork #makeup #redlip #oldhollywood #velvet #gold #sophistication #roses #romance #transformation #spotlight #confidence #selfcare #beauty #magic #style #fashion #gown #portrait #photography #photooftheday #instafashion #fashionista #ootd #eveningwear"""},
            {"role": "user", "content": """In a sprawling, lavender field, a model (Sydney Sweeney, Gigi Hadid) twirls with carefree abandon, her white, flowing dress billowing in the warm, summer breeze. The soft, purple hues of the lavender stretch out as far as the eye can see, creating a dreamlike, enchanted landscape that seems to have sprung straight from the pages of a fairytale.

        The model's hair is adorned with a delicate, floral crown, and her bare feet dance through the fragrant blooms, her laughter carrying across the field like a joyful melody. The golden, late afternoon sun casts a warm, ethereal glow across the scene, bathing the model in a soft, radiant light that seems to emanate from within.

        This whimsical, dreamlike portrait captures the essence of pure, unbridled joy, celebrating the beauty and magic of nature and the carefree spirit of youth. It is a reminder to embrace the simple, fleeting moments of happiness that color our lives, to dance with abandon amidst the blooms, and to let the warmth of the sun and the beauty of the world around us fill our hearts with wonder and delight."""},
            {"role": "assistant", "content": """В лавандовых полях, где мечты расцветают... 💜🌸
        .
        .
        .
        #lavender #field #whimsical #dreamlike #enchanted #fairytale #floralcrown #dance #laughter #joy #nature #magic #youth #happiness #wonder #delight #fashion #style #dress #portrait #photography #photooftheday #instafashion #fashionista #ootd #summer #naturalbeauty #dreamy #ethereal"""},
            {"role": "user", "content": """In a cozy, rustic cabin, a model (Sydney Sweeney, Gigi Hadid) is captured in a moment of quiet reflection, curled up on a plush, fur rug before a crackling fireplace. She wears an oversized, chunky knit sweater in a rich, cream hue, its soft folds enveloping her in a cocoon of warmth and comfort. Her golden hair is pulled back in a loose, messy bun, and her face is bare, revealing a natural, radiant beauty.

        The cabin's interior is a study in rustic charm, with exposed wooden beams, a stone fireplace, and an array of cozy, inviting textures. The warm, flickering light of the fire casts a gentle glow across the model's face, highlighting the soft, dreamy expression in her eyes as she gazes into the dancing flames, lost in thought.

        This intimate, introspective portrait captures a moment of pure, unguarded vulnerability, celebrating the beauty and power of solitude and self-reflection. It is a reminder to cherish the quiet moments, to find comfort and solace in the simple pleasures of life, and to embrace the raw, authentic beauty that shines through when we allow ourselves to simply be, without pretense or artifice."""},
            {"role": "assistant", "content": """В тишине мыслей, где душа находит уют... 🔥💭
        .
        .
        .
        #rustic #cabin #cozy #furrug #fireplace #knitsweater #natural #radiant #beauty #wooden #stone #warmth #comfort #solitude #selfreflection #vulnerability #authenticity #simplepleasures #introspective #fashion #style #portrait #photography #photooftheday #instafashion #fashionista #ootd #fallstyle #comfy #peaceful"""},
            {"role": "user", "content": """On a misty, ethereal morning, a model (Sydney Sweeney, Gigi Hadid) practices yoga on a serene, secluded beach. She wears a sleek, black yoga ensemble that hugs her toned, graceful figure, and her raven hair is pulled back into a sleek, high ponytail. Her pose is one of perfect balance and alignment, her body forming a graceful silhouette against the soft, gray sky.

        The gentle waves of the ocean lap at the shore, their rhythmic whispers a soothing soundtrack to her meditative practice. The cool, damp sand beneath her feet grounds her, while the salty breeze carries away any lingering thoughts or worries. The model's expression is one of pure, focused serenity, her eyes closed and her face tilted skyward, as if in reverence to the beauty and tranquility of the moment.

        This peaceful, meditative portrait captures the essence of mindfulness and the power of connecting with nature and oneself. It is a reminder to find stillness amidst the chaos, to cultivate inner peace and balance, and to embrace the simple, profound beauty of the present moment, one breath at a time."""},
            {"role": "assistant", "content": """В объятиях тишины, где йога встречает рассвет... 🧘‍♀️🌅
        .
        .
        .
        #yoga #beach #misty #ethereal #serene #secluded #balance #alignment #graceful #waves #ocean #meditative #serenity #mindfulness #innerpeace #nature #connection #stillness #chaos #beauty #presentmoment #breath #fashion #style #portrait #photography #photooftheday #instafashion #fashionista #ootd #yogawear #zen"""},
            {"role": "user", "content": photo_description}
        ]

        message = await self.client.messages.create(
            system=system_prompt,
            max_tokens=1024,
            messages=messages,
            model="claude-3-haiku-20240307",
        )

        caption = message.content[0].text

        hashtags = caption.split("#")
        caption = hashtags[0]
        hashtags = [f"#{hashtag.strip()}" for hashtag in hashtags[1:]]

        if len(hashtags) > 30:
            hashtags = hashtags[:30]

        return caption + " ".join(hashtags)

    @staticmethod
    async def generate_post_caption(photo_description: str) -> str:
        return await workflow.execute_activity(
            PhotoPromptGenerator._generate_post_caption,
            args=[photo_description],
            start_to_close_timeout=timedelta(seconds=30)
        )




