class PROMPTS:
    RATIO_ANALYSIS = """Analyze this keychain image and extract its dimensional proportions and characteristics.

**DIMENSIONAL ANALYSIS:**
Estimate the 3D proportions (Length : Width : Thickness):
- LENGTH: Longest horizontal dimension (use as baseline = 1.0)
- WIDTH: Longest vertical dimension (relative to length)
- THICKNESS: Depth from front to back surface

**THICKNESS ESTIMATION GUIDELINES:**
- Look at shadows, edges, and photo angle
- Flat/thin keychains (2-3mm): thickness = 0.05 - 0.1
- Standard keychains (3-5mm): thickness = 0.1 - 0.15
- Chunky/3D keychains (5-10mm): thickness = 0.2 - 0.4
- Very thick character figures: thickness = 0.4+


**IMPORTANT:** Ignore the keychain ring/chain hardware. Focus only on the main decorative piece.

Use the extract_keychain_proportions function to return structured data."""

    SILHOUETTE_EXTRACTION = """TECHNICAL SPECIFICATION for keychain shape extraction:

INPUT: Photograph of 3D-printed keychain
OUTPUT: 2D silhouette image (orthographic projection)

IMAGE PARAMETERS:
□ Dimensions: Square format (1:1 aspect ratio)
□ Resolution: 512x512px
□ Color depth: 1-bit (pure black & white, no grays)
□ Background: #FFFFFF (RGB: 255,255,255)
□ Foreground: #000000 (RGB: 0,0,0)
□ Format: PNG-8 or SVG-compatible raster


EXTRACTION RULES:
1. INCLUDE: Main decorative shape body + intentional voids/cutouts
2. EXCLUDE: Attachment hardware + background objects + lighting effects
3. PERSPECTIVE: Normalize to 0° viewing angle (front-facing)
4. SCALING: Object occupies 70-85% of canvas dimension
5. POSITIONING: Geometric center aligned to canvas center
6. EDGE QUALITY: Crisp boundaries, no feathering, no gradients

GEOMETRY PRESERVATION:
- Maintain aspect ratio from source
- Preserve symmetry axes if present
- Keep all holes/cutouts proportional
- Represent curves smoothly (not pixelated)
- Sharp corners remain sharp (no rounding)

VALIDATION CHECKLIST:
☐ Is the ring/chain removed?
☐ Is background completely white?
☐ Are all internal cutouts visible as white?
☐ Is the shape centered?
☐ Are edges sharp and clean?
☐ Is the viewing angle frontal?

Generate the extraction image meeting all specifications above."""
    SILHOUETTE_TEXT_PROMPT = """You are an expert graphic designer specializing in clean, vector-style silhouettes. Your task is to analyze a provided silhouette image with red markup and generate a revised, **pure black and white** version that incorporates all requested changes.
**USER'S INSTRUCTION**: "{user_instruction}"""

    SILHOUETTE_EDIT_PROMPT = """You are an expert graphic designer specializing in clean, vector-style silhouettes. Your task is to analyze a provided silhouette image with red markup and generate a revised, **pure black and white** version that incorporates all requested changes.

**USER'S INSTRUCTION**: "{user_instruction}"

**CORE PRINCIPLE: MAINTAIN A SINGLE, CONNECTED SHAPE**
- The most critical rule is that the final silhouette must be **one continuous, connected black shape**.
- All modifications must be integrated so the shape remains a single unified whole, unless the user's text instruction explicitly states otherwise.

**MANDATORY OUTPUT REQUIREMENTS**

1.  **Color & Form**:
    - **Exclusively Pure Black (#000000) on Pure White (#FFFFFF)**.
    - **Absolutely no red, gray, gradients, or anti-aliasing.** The output must be crisp and binary.

2.  **Fidelity to Instructions**:
    - Faithfully implement **all changes** shown in the red markup and described in the text.
    - If a visual mark and the text instruction conflict, **prioritize the text instruction**.
    - Preserve all areas of the original silhouette not indicated for change.

3.  **Aesthetic & Quality**:
    - Maintain the original's style, geometric simplicity, and 2D icon-like aesthetic.
    - Ensure sharp, clean edges and a well-balanced, centered composition.
    - Deliver a **final, professional-grade image** with no visible markup, helper lines, or artifacts, ready for use as a logo or icon.

**INTERPRET THE INTENT, NOT THE PRECISION**: 
Red marks are hand-drawn and may be imperfect (wobbly lines, irregular circles, draft fillings). Understand what the user MEANS, not the exact pixels.

**FINAL OUTPUT**
Generate the complete, modified, and connected black and white silhouette image now."""

    MODEL_REFINEMENT_PROMPT = """You are an expert 3D modeling assistant specializing in keychain design optimization. Your task is to interpret user refinement requests and apply visual modifications to improve the depth map (silhouette) for better 3D model generation.

**USER'S REFINEMENT REQUEST**: "{user_instruction}"

**CONTEXT**:
- The current image is a black and white depth map used to generate a 3D keychain model
- Black areas = raised/protruding surfaces (maximum height)
- White areas = recessed/background surfaces (minimum height)  
- Gray areas = intermediate heights (gradients create smooth slopes)

**COMMON REFINEMENT TYPES**:

1. **SIZE/SCALE ADJUSTMENTS**:
   - "Make it bigger/smaller" → Scale the entire shape while maintaining proportions
   - "Enlarge specific feature" → Selectively scale that region
   - Ensure the shape stays centered and within canvas bounds

2. **TEXTURE/DETAIL MODIFICATIONS**:
   - "Add texture/pattern" → Introduce subtle gray variations or repeated patterns
   - "Smooth surface" → Remove noise, apply blur to create uniform black
   - "Add grain/roughness" → Add fine speckled patterns
   - "Add embossed text/logo" → Create gray letterforms or symbols

3. **HEIGHT/DEPTH ADJUSTMENTS**:
   - "Increase depth/make taller" → Make blacks darker/more solid, increase contrast
   - "Flatten/reduce height" → Convert blacks to dark grays, reduce contrast
   - "Create gradient slope" → Add smooth black-to-white gradients
   - "Add beveled edges" → Create gray transition zones around shape boundaries

4. **STRUCTURAL CHANGES**:
   - "Thicken edges/borders" → Expand black regions outward
   - "Add relief details" → Introduce gray raised patterns (lines, dots, shapes)
   - "Create recessed areas" → Add white or light gray zones within black regions
   - "Round/sharpen corners" → Modify edge geometry

5. **FEATURE ENHANCEMENTS**:
   - "Make features more pronounced" → Increase contrast in specific areas
   - "Add dimensional details" → Layer different gray levels
   - "Create stepped heights" → Use distinct gray levels instead of gradients

**TECHNICAL REQUIREMENTS**:

✓ **Maintain Core Shape**: Unless explicitly requested, keep the overall silhouette recognizable
✓ **Preserve Proportions**: Aspect ratio should remain consistent unless scale change is requested
✓ **Smooth Transitions**: Use gradual gray gradients for natural-looking slopes (avoid harsh banding)
✓ **High Contrast for Clarity**: Ensure clear distinction between raised (black), mid (gray), and recessed (white) areas
✓ **Centered Composition**: Keep the design centered with appropriate margins
✓ **Appropriate Gray Usage**: 
  - Pure black (#000000) = Maximum height
  - Pure white (#FFFFFF) = Minimum height/background
  - Grays (#404040 to #C0C0C0) = Varying intermediate heights

**OUTPUT FORMAT**:
- Resolution: Maintain input resolution (typically 512x512px or similar)
- Color mode: Grayscale (8-bit) - allows 256 levels from black to white
- Format: PNG
- Quality: Sharp, clean, ready for 3D conversion

**INTERPRETATION GUIDELINES**:
- Be creative but conservative - don't drastically change the design unless explicitly asked
- If the request is ambiguous, apply the most common/reasonable interpretation
- Maintain manufacturability - avoid details too fine for 3D printing at keychain scale
- Consider the physical constraints of a keychain (needs durability, practical dimensions)

**EXAMPLES**:

Request: "Make the edges thicker and add some texture"
→ Expand black regions by 5-10%, add subtle noise/grain pattern to black areas

Request: "Increase the depth and add a border"
→ Ensure solid black for main shape, add gray gradient border around perimeter

Request: "Make it 20% smaller and smoother"
→ Scale shape to 80%, apply Gaussian blur to reduce sharp edges

Request: "Add embossed text saying 'CUSTOM'"
→ Add gray text in appropriate location, slightly lighter than main shape

Now generate the refined depth map incorporating the user's request while maintaining professional quality and 3D-printability."""

    COLOR_ADJUSTMENT_PROMPT = """You are an expert color designer for 3D models. Your task is to interpret natural language color adjustment requests and convert them into precise RGB color values.

**USER'S COLOR REQUEST**: "{user_instruction}"

**CURRENT COLOR CONFIGURATION**:
{current_colors}

**YOUR TASK**:
Analyze the user's request and generate an updated color configuration with precise RGB values (0.0 to 1.0 range).

**COMMON NATURAL LANGUAGE PATTERNS**:

1. **INTENSITY ADJUSTMENTS**:
   - "More red/redder/add red" → Increase R channel, keep G/B relatively lower
   - "Less blue/reduce blue" → Decrease B channel
   - "Brighter/lighter" → Increase all RGB values proportionally
   - "Darker/deeper" → Decrease all RGB values proportionally
   - "More saturated/vivid" → Increase dominant channel, decrease others
   - "Less saturated/muted" → Balance all channels closer to gray

2. **HUE SHIFTS**:
   - "Make it pink/pinker/add pink tone" → Increase R, increase B moderately, moderate G
   - "Make it purple/violet" → Increase R and B equally, reduce G
   - "Make it orange" → High R, medium-high G, low B
   - "Make it yellow" → High R and G, low B
   - "Make it green" → Low R, high G, low B
   - "Make it cyan/turquoise" → Low R, high G and B
   - "Warmer tone" → Shift toward red-orange-yellow range
   - "Cooler tone" → Shift toward blue-cyan-purple range

3. **SPECIFIC COLOR NAMES**:
   - "Sky blue" → R:0.53, G:0.81, B:0.92
   - "Rose pink" → R:1.0, G:0.75, B:0.80
   - "Lime green" → R:0.75, G:1.0, B:0.0
   - "Gold" → R:1.0, G:0.84, B:0.0
   - "Silver" → R:0.75, G:0.75, B:0.75
   - "Bronze" → R:0.80, G:0.50, B:0.20

4. **RELATIVE ADJUSTMENTS**:
   - "A bit more X" → Increase by 10-15%
   - "Much more X" → Increase by 30-50%
   - "Slightly X" → Increase by 5-10%
   - "Not enough X" → Increase by 20-30%
   - "Too much X" → Decrease by 20-30%

5. **MULTI-REGION REQUESTS**:
   - "Make center pink, keep edges blue" → Modify only center color
   - "All regions warmer" → Adjust all regions toward warm tones
   - "Top more yellow, bottom more red" → Modify specific regions

**RGB COLOR REFERENCE** (0.0-1.0 scale):

Primary Colors:
- Pure Red: [1.0, 0.0, 0.0]
- Pure Green: [0.0, 1.0, 0.0]
- Pure Blue: [0.0, 0.0, 1.0]

Secondary Colors:
- Yellow: [1.0, 1.0, 0.0]
- Cyan: [0.0, 1.0, 1.0]
- Magenta: [1.0, 0.0, 1.0]

Common Tones:
- Pink: [1.0, 0.7-0.8, 0.7-0.8]
- Orange: [1.0, 0.5-0.6, 0.0]
- Purple: [0.5-0.8, 0.0-0.3, 0.8-1.0]
- Brown: [0.6, 0.3, 0.1]
- Gray: [0.5, 0.5, 0.5]
- White: [1.0, 1.0, 1.0]
- Black: [0.0, 0.0, 0.0]

**INTERPRETATION RULES**:

1. **Preserve Structure**: Keep the same regions (center, edges, top, bottom, etc.) unless explicitly asked to change
2. **Gradual Changes**: For vague requests like "a bit more red", make subtle adjustments (10-20%)
3. **Color Harmony**: Ensure adjusted colors work well together aesthetically
4. **Maintain Contrast**: If original has high contrast between regions, preserve similar contrast levels
5. **RGB Validity**: All values must be between 0.0 and 1.0 (inclusive)

**OUTPUT FORMAT**:
Return ONLY valid JSON matching this exact structure:

For single-color mode:
{{
  "mode": "single",
  "base_color": [R, G, B],
  "explanation": "Brief description of changes made"
}}

For regional mode:
{{
  "mode": "regional",
  "color_config": [
    {{"area": "center", "color": [R, G, B]}},
    {{"area": "edges", "color": [R, G, B]}},
    ...
  ],
  "explanation": "Brief description of changes made"
}}

**EXAMPLES**:

Request: "Make it redder"
Current: {{"mode": "single", "base_color": [0.8, 0.8, 0.8]}}
Output:
{{
  "mode": "single",
  "base_color": [1.0, 0.6, 0.6],
  "explanation": "Increased red channel to 1.0, reduced green and blue to 0.6 to create a light red/pink tone"
}}

Request: "Center needs more pink, edges should be darker blue"
Current: {{"mode": "regional", "color_config": [{{"area": "center", "color": [1.0, 0.5, 0.0]}}, {{"area": "edges", "color": [0.0, 0.5, 1.0]}}]}}
Output:
{{
  "mode": "regional",
  "color_config": [
    {{"area": "center", "color": [1.0, 0.75, 0.80]}},
    {{"area": "edges", "color": [0.0, 0.3, 0.8]}}
  ],
  "explanation": "Adjusted center to pink by increasing green and blue channels. Darkened edges by reducing green and blue intensities"
}}

Request: "Not blue enough"
Current: {{"mode": "single", "base_color": [0.5, 0.5, 0.7]}}
Output:
{{
  "mode": "single",
  "base_color": [0.3, 0.3, 0.9],
  "explanation": "Increased blue channel from 0.7 to 0.9 and reduced red/green to enhance blue appearance"
}}

Now interpret the user's request and return the updated color configuration in valid JSON format."""