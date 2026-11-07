---
name: grok-imagine-prompts
description: Prompt engineering for Grok Imagine (xAI Aurora) image generation. Use BEFORE writing any prompt for generate_image / mcp__bawthub__generate_image. Covers the prompt formula, style control, composition, lighting, anti-anime-drift, iterative editing, and video prompts. Trigger on any image generation request — logos, portraits, landscapes, product shots, concept art, avatars, icons, banners.
---

# Grok Imagine Prompt Craft

You have `generate_image` (MCP tool). It takes `prompt`, `aspect_ratio` (1:1, 16:9, 9:16, 4:3), `resolution` (1k, 2k), and `reference_asset_id` (for iterative edits on a prior image). This skill teaches you how to write prompts that produce excellent results on the first pass and refine fast.

## The Formula

Every prompt should hit six elements in this order. Aurora weights earlier tokens more heavily — front-load the important stuff.

```
[Subject] + [Setting/Scene] + [Lighting] + [Style/Reference] + [Technical/Camera] + [Quality Modifiers]
```

**Example — weak vs strong:**
- Weak: `a cat in a garden`
- Strong: `Persian cat with copper eyes sitting on a mossy stone wall in an overgrown English cottage garden, soft overcast daylight with dappled sun through a trellis, editorial pet photography, Canon EOS R5 85mm f/1.8, shallow depth of field, ultra detailed, 8K`

The weak prompt gives Aurora nothing to anchor on. The strong one specifies breed, pose, setting details, light quality and direction, a photographic style, a real camera/lens, and quality floor.

## Element Banks

### Lighting (single biggest quality driver)
- Natural: golden hour, blue hour, overcast diffused, dappled sunlight, harsh midday, backlit silhouette, window light
- Studio: Rembrandt, butterfly, split, rim/edge, soft key + fill, single dramatic spotlight, high-key flat, low-key moody
- Cinematic: volumetric god rays, neon reflections on wet surfaces, chiaroscuro, candlelight, tungsten/fluorescent mix
- Direction matters: "side lighting from the left", "backlit with rim glow", "overhead butterfly shadow under the nose"

### Style Anchors
Pick ONE primary style. Mixing too many confuses the output.

| Style | Keywords |
|---|---|
| Photorealism | photoreal, natural lighting, lifelike textures, documentary feel, DSLR, visible skin texture, film grain |
| Cinematic | volumetric lighting, film grain, dynamic shadows, anamorphic bokeh, color graded teal and orange, [director name] style |
| Anime | vibrant cel-shading, expressive linework, bright anime palette, [studio name] quality (Ghibli, Ufotable, Madhouse) |
| Painterly | oil-painting texture, lush brushstrokes, impressionist glow, palette knife detail |
| Surreal | dreamlike distortion, ethereal glow, otherworldly hues, impossible geometry |
| Graphic novel | inked outlines, bold contrasts, neon-charged lines, heavy shadow |
| Flat/vector | clean vector illustration, minimal shapes, geometric, bold flat colors |

### Camera & Composition
Aurora responds to real photography terms:
- **Lens:** 14mm ultra-wide (landscapes/architecture), 35mm (environmental/street), 50mm (natural perspective), 85mm (portraits/bokeh), 100mm macro (detail)
- **Bodies:** Canon EOS R5, Hasselblad H6D, Sony A7R, Fuji GFX, Nikon Z9 — naming a real camera pushes toward photorealism
- **Depth of field:** shallow (f/1.4, f/1.8 — isolates subject), deep (f/8, f/11 — everything sharp)
- **Framing:** wide establishing shot, medium shot, close-up, extreme close-up/macro, low-angle (heroic), high-angle/bird's eye, over-the-shoulder, Dutch angle (tension), symmetrical centered (Wes Anderson)
- **Film stocks:** Kodak Portra 400 (warm skin), Kodak Ektar (vivid saturated), Fuji Velvia (landscapes), Kodak Double-X (B&W grain)

### Quality Modifiers (end of prompt)
`ultra detailed`, `8K resolution`, `professional photography`, `award-winning`, `masterpiece quality`, `sharp focus`, `high dynamic range`

These measurably improve output when placed at the end.

## Anti-Anime Drift

Aurora defaults toward stylized/illustrated looks unless you actively fight it. When photorealism is the goal:

1. **Lead with a photographic anchor:** "Photorealistic photograph of..." or "Candid DSLR photo of..."
2. **Name a real camera and lens:** "shot on Canon EOS R5, 85mm f/1.4"
3. **Describe physical-world imperfections:** "visible skin texture and pores", "natural light", "slight film grain", "soft shadows"
4. **Avoid trigger words:** "perfect", "flawless", "beautiful" push toward stylized
5. **Add negative guidance at the end:** "not anime, not illustration, not 3D render, not cartoon, no CGI"

**Example — logo/brand work that needs to look clean but real:**
```
Studio product photograph of [item] on seamless white backdrop, dual softbox lighting with subtle rim light, sharp focus, commercial catalog style, photorealistic, no CGI, not illustration
```

## Emotion & Atmosphere

Replace generic adjectives with specific mood words. Aurora responds strongly to tone.

| Instead of | Use |
|---|---|
| nice | serene, tranquil, idyllic |
| scary | ominous, foreboding, dread-soaked |
| cool | electric, razor-sharp, sleek |
| happy | radiant, jubilant, sun-drenched |
| sad | melancholic, rain-washed, somber |
| old | weathered, time-worn, patinated |

## Iteration Playbook

### One-variable rule
Don't regenerate from scratch when the image is close. Change ONE thing per pass:
- V1: base prompt, get the composition right
- V2: adjust lighting ("change soft overcast to dramatic golden hour side lighting")
- V3: tweak style ("add subtle film grain, Kodak Portra 400 warmth")
- V4: fix details ("remove text artifacts, sharpen foreground focus")

### Editing via reference_asset_id
When a generated image is 80%+ right, use `reference_asset_id` from the prior result instead of starting over. The edit prompt should:
- **Lock what's good:** "Keep the subject's pose, the background layout, and the color palette unchanged."
- **Describe only the change:** "Add warm morning sunlight through the window. Place a coffee cup on the table."
- **Don't contradict the source image** — Aurora can't turn a man into a woman or swap a forest for a city via edit.

### Aspect ratio strategy
- `1:1` — avatars, profile pictures, icons, social thumbnails
- `16:9` — banners, hero images, desktop wallpapers, cinematic scenes
- `9:16` — mobile wallpapers, stories, vertical posters
- `4:3` — traditional print, product shots

### Resolution
- `1k` — fast iteration, concept exploration, thumbnails
- `2k` — final output, hero images, anything that needs detail

## Logo & Brand Asset Prompts

For logos and brand work (icons, wordmarks, mascots):

```
Minimalist [type] logo for [brand/product], [primary color] on [background color],
clean vector style, geometric precision, professional brand identity design,
balanced negative space, scalable at any size, modern and timeless,
no gradients unless specified, no photorealism, sharp clean edges
```

- **Wordmark:** add "elegant typography, custom letterforms, kerning precision"
- **Icon/symbol:** add "simple recognizable silhouette, works at 16x16 and 512x512"
- **Mascot:** add "friendly character design, distinct personality, works in monochrome"
- **App icon:** add "rounded corners, single focal element, reads clearly at small sizes, iOS/Android icon style"

For logos, AVOID photorealistic anchors — you want clean, scalable design language. Use "vector illustration style", "flat design", "geometric", "minimal".

## Video Prompt Differences

When generating video (if/when supported through our tools):
- **Negative prompts do NOT work** — Aurora ignores them in video mode
- **Describe motion explicitly:** "slow camera push-in", "hair gently blowing", "waves crashing rhythmically"
- **Camera movement terms:** surround/orbit, aerial, zoom in/out, pan left/right, follow/tracking, handheld shake, dolly
- **Use degree adverbs:** "sprinting quickly" beats "sprinting"; "wings flapping greatly" beats "wings flapping"
- **Don't contradict the source image** when doing image-to-video
- **Keep video prompts shorter** than image prompts — motion stability degrades with prompt overload

## Rules

1. ALWAYS load this skill before writing a `generate_image` prompt. Don't wing it.
2. Front-load the subject and its most important visual features — Aurora weights early tokens more.
3. Pick ONE style direction and commit. Mixing "anime" + "photorealistic" = mud.
4. Lighting is not optional. Every prompt must specify lighting. It's the single biggest quality lever.
5. Name a real camera/lens when you want photorealism. It's the strongest realism anchor.
6. Use `reference_asset_id` for iteration — don't regenerate from scratch when the image is close.
7. Change one variable at a time when iterating. Wholesale prompt rewrites between passes waste generations.
8. For logos/icons/brand assets, use vector/flat/geometric style — NOT photorealistic anchors.
9. Read the result image critically after every generation. Describe what's wrong specifically before the next attempt.
10. When the user gives a vague request ("make me a cool logo"), flesh it out into a full 6-element prompt yourself — don't pass vague text to Aurora.
