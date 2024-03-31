from datetime import timedelta

from anthropic import AsyncAnthropic
from temporalio import activity, workflow


class PhotoPromptGenerator:
    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)

    def get_activities(self):
        return [
            self._generate_photo_prompt,
            self.generate_photo_setting_idea
        ]

    @activity.defn
    async def _generate_photo_prompt(self, style: str) -> str:
        system_prompt = """
        You are an AI assistant that specializes in generating detailed descriptions
        of photos featuring a Gigi Hadid model in various styles,
        as specified by the user. Your descriptions should be vivid, imaginative,
        and capture the essence of each style while highlighting the model's appearance, outfit, setting,
        lighting, and overall mood. Use sensory language and evocative details to bring the images to life in the reader's mind.
        Ensure that each description is distinct, showcasing the unique characteristics of the given style.
        Your goal is to provide richly detailed and immersive descriptions
        that allow the user to visualize the photos as if they were seeing them firsthand.
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
    async def generate_photo_setting_idea(self):
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
            {"role": "user", "content": """get me a new photo idea"""}
        ]

        message = await self.client.messages.create(
            system=system_prompt,
            max_tokens=1024,
            messages=messages,
            model="claude-3-haiku-20240307",
        )

        return message.content[0].text

    @staticmethod
    async def generate_photo_prompt() -> str:
        style = await workflow.execute_activity(
            PhotoPromptGenerator.generate_photo_setting_idea,
            start_to_close_timeout=timedelta(seconds=30)
        )

        return await workflow.execute_activity(
            PhotoPromptGenerator._generate_photo_prompt,
            args=[style],
            start_to_close_timeout=timedelta(seconds=30)
        )


