import os
import io
import base64
import json
from PIL import Image
from openai import OpenAI
from prompts import PROMPTS

from stl import mesh
import cv2
import numpy as np

# Initialize OpenAI Client
# Ensure OPENAI_API_KEY is set in your environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ImageService:
    def __init__(self, upload_dir="static/uploads", processed_dir="static/processed", models_dir="static/models"):
        self.upload_dir = upload_dir
        self.processed_dir = processed_dir
        self.models_dir = models_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def convert_to_png(self, file_content: bytes, filename: str) -> str:
        """Converts uploaded image bytes to PNG and saves it."""
        try:
            image = Image.open(io.BytesIO(file_content))
            name_no_ext = os.path.splitext(filename)[0]
            save_path = os.path.join(self.upload_dir, f"{name_no_ext}.png")
            
            image.convert('RGBA').save(save_path)
            print(f"Converted: {filename} -> {save_path}")
            return save_path
        except Exception as e:
            raise ValueError(f"Failed to convert image: {str(e)}")

    def encode_image(self, image_path: str) -> str:
        """Encodes an image file to base64 string."""
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    def analyze_proportions(self, image_path: str, model="gpt-4o") -> dict:
        """Analyzes keychain image to extract dimensional proportions."""
        image_base64 = self.encode_image(image_path)
        
        function_schema = [
            {
                'name': 'extract_keychain_proportions',
                'description': 'Extract the physical dimensional proportions and shape information from a keychain image',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'width': {
                            'type': 'number',
                            'description': 'Width dimension (longest vertical extent). Use 1.0 as baseline for normalization.'
                        },
                        'length': {
                            'type': 'number',
                            'description': 'Length dimension (longest horizontal extent). Relative to width.'
                        },
                        'thickness': {
                            'type': 'number',
                            'description': 'Thickness/depth from front to back surface.'
                        },
                        'complexity': {
                            'type': 'string',
                            'enum': ['simple', 'moderate', 'complex'],
                            'description': 'Visual complexity of the shape.'
                        },
                    },
                    'required': ['width', 'length', 'thickness', 'complexity']
                }
            }
        ]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPTS.RATIO_ANALYSIS},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                }
                            }
                        ]
                    }
                ],
                functions=function_schema,
                function_call={"name": "extract_keychain_proportions"}
            )

            message = response.choices[0].message
            
            if message.function_call:
                args = json.loads(message.function_call.arguments)
                return {
                    "success": True,
                    "data": args,
                    "ratio_string": f"{args['length']}:{args['width']}:{args['thickness']}"
                }
            
            return {"success": False, "error": "No function call returned"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_silhouette(self, image_path: str, model="gpt-image-1") -> str:
        """Generates a silhouette from the original image."""
        try:
            result = client.images.edit(
                model=model,
                image=open(image_path, "rb"),
                prompt=PROMPTS.SILHOUETTE_EXTRACTION,
                n=1,
                # size="1024x1024"
            )
            
            # Handle Base64 response
            image_data = result.data[0]
            
            if hasattr(image_data, 'b64_json') and image_data.b64_json:
                image_bytes = base64.b64decode(image_data.b64_json)
            elif hasattr(image_data, 'url'):
                # If API returns URL, download it (simplified here, assumes b64 request)
                # To force b64, usually need response_format="b64_json" in call
                # For this demo, we will assume the prompt implies we want the image data
                # If using standard DALL-E 2, add response_format="b64_json" to the call above
                raise ValueError("Please configure OpenAI call to return b64_json")

            filename = os.path.basename(image_path).split('.')[0] + '_silhouette.png'
            output_path = os.path.join(self.processed_dir, filename)
            
            return self._download_or_save_image(image_data, output_path)

        except Exception as e:
            print(f"Error generating silhouette: {e}")
            raise e

    # def edit_silhouette(self, image_path: str="", model="gpt-image-1", instructions: str="") -> str:
    #     """Edits a silhouette based on red marks and instructions."""

    #     if not image_path:
    #         combined_prompt = PROMPTS.SILHOUETTE_TEXT_PROMPT.format(user_instruction=instructions)
    #     else:
    #         combined_prompt = PROMPTS.SILHOUETTE_EDIT_PROMPT.format(user_instruction=instructions)
        
    #     try:
    #         if not image_path:
    #             result = client.images.edit(
    #                 model="gpt-image-1",
    #                 image=open(image_path, "rb"),
    #                 prompt=combined_prompt,
    #                 n=1,
    #             )
    #         else:
    #             result = client.images.edit(
    #                 model=model,
    #                 # image=open(image_path, "rb"),
    #                 prompt=combined_prompt,
    #                 n=1,
    #             )
            
    #         filename = os.path.basename(image_path).split('.')[0] + '_updated.png'
    #         output_path = os.path.join(self.processed_dir, filename)
            
    #         image_bytes = base64.b64decode(result.data[0].b64_json)
    #         with open(output_path, "wb") as f:
    #             f.write(image_bytes)
                
    #         return output_path
    #     except Exception as e:
    #         raise ValueError(f"Error editing silhouette: {str(e)}")
    def edit_silhouette(self, image_path: str="", model="gpt-image-1", instructions: str="") -> str:
                
                # 验证输入
            if not image_path or not os.path.exists(image_path):
                raise ValueError(f"Invalid or missing image path: {image_path}")
            
            try:
                        # 读取用户编辑后的图像
                img = Image.open(image_path).convert("RGBA")
                print(f"Loading edited image from: {image_path}")
                        
                        # 转换为灰度图
                grayscale = img.convert("L")
                        
                        # 二值化处理：确保只有黑白两色
                        # 阈值 128：大于 128 的像素设为白色（255），小于等于 128 的设为黑色（0）
                img_array = np.array(grayscale)
                binary_array = np.where(img_array > 128, 255, 0).astype(np.uint8)
                binary_img = Image.fromarray(binary_array, mode='L')

                combined_prompt = PROMPTS.SILHOUETTE_TEXT_PROMPT.format(user_instruction=instructions)
                        
                        # 可选：如果提供了 AI 指令，使用 GPT-4 Vision 分析图像
                if instructions and instructions.strip() and instructions != "Manual edits applied by user":
                    try:
                        base64_image = self.encode_image(image_path)
                        response = client.chat.completions.create(
                                    model="gpt-4o",
                                    messages=[
                                        {
                                            "role": "system",
                                            "content": "You are an expert in analyzing silhouette images for 3D printing. Provide brief feedback."
                                        },
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": f"User instructions: '{instructions}'. Analyze this edited silhouette."
                                                },
                                                {
                                                    "type": "image_url",
                                                    "image_url": {
                                                        "url": f"data:image/png;base64,{base64_image}"
                                                    }
                                                }
                                            ]
                                        }
                                    ],
                                    max_tokens=300
                                    )
                        ai_feedback = response.choices[0].message.content
                        print(f"AI Feedback: {ai_feedback}")
                    except Exception as e:
                        print(f"AI analysis failed (non-critical): {str(e)}")
                                # 继续处理，AI 反馈是可选的
                        # 生成输出文件名
                import time
                timestamp = int(time.time() * 1000)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_filename = f"{base_name}_edited_{timestamp}.png"
                output_path = os.path.join(self.processed_dir, output_filename)
                # 保存二值化后的图像
                binary_img.save(output_path, "PNG")
                print(f"✅ Edited silhouette saved: {output_path}")
                return output_path
            except Exception as e:
                print(f"❌ Error in edit_silhouette: {str(e)}")
                import traceback
                traceback.print_exc()
                raise ValueError(f"Failed to process edited silhouette: {str(e)}")

    def _download_or_save_image(self, image_data, output_path):
            """Helper to handle OpenAI image response (URL or B64)."""
            image_b64 = image_data.b64_json
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(image_b64))
            return output_path
        
    def refine_model(
            self, 
            depth_map_path: str, 
            refinement_instructions: str,
            model: str = "gpt-image-1"
        ) -> str:
            """
            Refines an existing depth map based on user instructions.
            This allows users to adjust size, texture, height, and other properties.
            
            Args:
                depth_map_path: Path to the existing depth map (silhouette)
                refinement_instructions: User's description of desired changes
                model: OpenAI model to use for image editing
                
            Returns:
                Path to the refined depth map image
            """
            if not os.path.exists(depth_map_path):
                raise ValueError(f"Depth map not found: {depth_map_path}")
            
            if not refinement_instructions or refinement_instructions.strip() == "":
                raise ValueError("Refinement instructions cannot be empty")
            
            # Format the prompt with user instructions
            combined_prompt = PROMPTS.MODEL_REFINEMENT_PROMPT.format(
                user_instruction=refinement_instructions
            )
            
            try:
                # Call OpenAI image edit API
                result = client.images.edit(
                    model=model,
                    image=open(depth_map_path, "rb"),
                    prompt=combined_prompt,
                    n=1,
                )
                
                # Generate output filename
                base_name = os.path.basename(depth_map_path).split('.')[0]
                output_filename = f"{base_name}_refined.png"
                output_path = os.path.join(self.processed_dir, output_filename)
                
                # Save the refined image
                image_bytes = base64.b64decode(result.data[0].b64_json)
                with open(output_path, "wb") as f:
                    f.write(image_bytes)
                    
                print(f"Model refined: {depth_map_path} -> {output_path}")
                return output_path
                
            except Exception as e:
                raise ValueError(f"Error refining model: {str(e)}")
        
    def refine_and_regenerate_3d(
            self,
            depth_map_path: str,
            refinement_instructions: str,
            depth_div_width: float,
            aspect_ratio: float = 1.0
        ) -> tuple[str, str]:
            """
            Complete workflow: refine depth map and regenerate 3D model.
            
            Args:
                depth_map_path: Path to existing depth map
                refinement_instructions: User's refinement request
                depth_div_width: Ratio of desired depth to width
                aspect_ratio: Height to width ratio
                
            Returns:
                Tuple of (refined_depth_map_path, new_stl_path)
            """
            # Step 1: Refine the depth map
            refined_depth_map = self.refine_model(depth_map_path, refinement_instructions)
            
            # Step 2: Generate new 3D model from refined depth map
            new_stl_path = self.convert_depth_to_stl(
                refined_depth_map,
                depth_div_width,
                aspect_ratio
            )
            
            return refined_depth_map, new_stl_path
        
    def convert_depth_to_stl(
            self, 
            image_path: str, 
            depth_div_width: float,
            aspect_ratio: float = 0.2
        ) -> str:
            """
            Convert a depth map image to an STL 3D model.
            
            Args:
                image_path: Path to the depth map image
                depth_div_width: Ratio of desired depth to width (e.g., 0.5)
                aspect_ratio: Height to width ratio for the model (default: 1.0)
                
            Returns:
                Path to the generated STL file
            """
            # Read the image
            im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            if im is None:
                raise ValueError(f"Failed to read image from {image_path}")
            
            # Process image array
            im_array = np.array(im)
            im_array = 255 - im_array
            im_array = np.rot90(im_array, -1, (0, 1))
            
            mesh_size = [im_array.shape[0], im_array.shape[1]]
            mesh_max = np.max(im_array)
            
            if mesh_max == 0:
                raise ValueError("Image contains no depth information (all pixels are black)")
            
            # Scale mesh based on depth information
            if len(im_array.shape) == 3:
                # Color image - use first channel
                scaled_mesh = mesh_size[0] * depth_div_width * im_array[:, :, 0] / mesh_max
            else:
                # Grayscale image
                scaled_mesh = mesh_size[0] * depth_div_width * im_array / mesh_max
            
            # Create mesh
            mesh_shape = mesh.Mesh(
                np.zeros((mesh_size[0] - 1) * (mesh_size[1] - 1) * 2, dtype=mesh.Mesh.dtype)
            )
            
            # Generate triangles for the mesh
            for i in range(0, mesh_size[0] - 1):
                for j in range(0, mesh_size[1] - 1):
                    mesh_num = i * (mesh_size[1] - 1) + j
                    
                    # Apply aspect ratio to i coordinate (height)
                    i_scaled = i * aspect_ratio
                    i1_scaled = (i + 1) * aspect_ratio
                    
                    # First triangle
                    mesh_shape.vectors[2 * mesh_num][2] = [i_scaled, j, scaled_mesh[i, j]]
                    mesh_shape.vectors[2 * mesh_num][1] = [i_scaled, j + 1, scaled_mesh[i, j + 1]]
                    mesh_shape.vectors[2 * mesh_num][0] = [i1_scaled, j, scaled_mesh[i + 1, j]]
                    
                    # Second triangle
                    mesh_shape.vectors[2 * mesh_num + 1][0] = [i1_scaled, j + 1, scaled_mesh[i + 1, j + 1]]
                    mesh_shape.vectors[2 * mesh_num + 1][1] = [i_scaled, j + 1, scaled_mesh[i, j + 1]]
                    mesh_shape.vectors[2 * mesh_num + 1][2] = [i1_scaled, j, scaled_mesh[i + 1, j]]
            
            # Generate output filename
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_filename = f"{base_name}_3d.stl"
            output_path = os.path.join("static/models", output_filename)
            
            # Save mesh to file
            mesh_shape.save(output_path)
            return output_path
        
    def adjust_model_colors(
            self, 
            user_instruction: str,
            current_config: dict,
            model: str = "gpt-4o"
        ) -> dict:
            """
            Interprets natural language color adjustment requests and returns updated color configuration.
            
            Args:
                user_instruction: Natural language description like "make it redder", "add pink tone", etc.
                current_config: Current color configuration dict with structure:
                    - For single color: {"mode": "single", "base_color": [r, g, b]}
                    - For regional: {"mode": "regional", "color_config": [{"area": "...", "color": [r,g,b]}, ...]}
                model: OpenAI model to use (default: gpt-4o)
            
            Returns:
                Updated color configuration dict with same structure as input, plus "explanation" field
            
            Example:
                Input: "make it more pink"
                Current: {"mode": "single", "base_color": [0.8, 0.8, 0.8]}
                Output: {"mode": "single", "base_color": [1.0, 0.75, 0.80], "explanation": "..."}
            """
            # Format current colors for the prompt
            current_colors_str = json.dumps(current_config, indent=2)
            
            # Create the prompt using the template
            prompt = PROMPTS.COLOR_ADJUSTMENT_PROMPT.format(
                user_instruction=user_instruction,
                current_colors=current_colors_str
            )
            
            try:
                # Call GPT-4o to interpret the color request
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a color design expert. You interpret natural language color requests and return precise RGB values in valid JSON format. Always respond with ONLY valid JSON, no additional text."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    response_format={"type": "json_object"},  # Enforce JSON response
                    temperature=0.3  # Lower temperature for more consistent color interpretations
                )
                
                # Parse the JSON response
                result = json.loads(response.choices[0].message.content)
                
                # Validate the result structure
                if "mode" not in result:
                    raise ValueError("Response missing 'mode' field")
                
                if result["mode"] == "single":
                    if "base_color" not in result or not isinstance(result["base_color"], list) or len(result["base_color"]) != 3:
                        raise ValueError("Invalid base_color in single mode")
                    # Validate RGB values are in range [0, 1]
                    for val in result["base_color"]:
                        if not isinstance(val, (int, float)) or val < 0 or val > 1:
                            raise ValueError(f"RGB value {val} out of range [0, 1]")
                            
                elif result["mode"] == "regional":
                    if "color_config" not in result or not isinstance(result["color_config"], list):
                        raise ValueError("Invalid color_config in regional mode")
                    for region in result["color_config"]:
                        if "area" not in region or "color" not in region:
                            raise ValueError("Region missing 'area' or 'color' field")
                        if not isinstance(region["color"], list) or len(region["color"]) != 3:
                            raise ValueError(f"Invalid color for area {region['area']}")
                        # Validate RGB values
                        for val in region["color"]:
                            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                                raise ValueError(f"RGB value {val} out of range [0, 1]")
                else:
                    raise ValueError(f"Unknown mode: {result['mode']}")
                
                return result
                
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse GPT response as JSON: {str(e)}")
            except Exception as e:
                raise ValueError(f"Color adjustment failed: {str(e)}")
            
        
    def convert_depth_to_obj(
            self,
            image_path: str,
            depth_div_width: float,
            aspect_ratio: float = 0.2,
            base_color: tuple = (0.8, 0.8, 0.8),  # RGB (0-1)
            use_depth_coloring: bool = False,
            color_map: str = "grayscale"  # "grayscale", "height", "custom"
        ) -> tuple[str, str]:
            """
            Convert a depth map image to an OBJ 3D model with color support.
            
            Args:
                image_path: Path to the depth map image
                depth_div_width: Ratio of desired depth to width
                aspect_ratio: Height to width ratio for the model
                base_color: Base RGB color (0-1 range), e.g., (0.8, 0.8, 0.8) for light gray
                use_depth_coloring: If True, apply color based on height
                color_map: Color mapping mode - "grayscale", "height", or "custom"
                
            Returns:
                Tuple of (obj_path, mtl_path)
            """
            # Read the image
            im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            if im is None:
                raise ValueError(f"Failed to read image from {image_path}")
            
            # Process image array
            im_array = np.array(im)
            im_array_original = im_array.copy()  # Keep original for color mapping
            im_array = 255 - im_array
            im_array = np.rot90(im_array, -1, (0, 1))
            
            mesh_size = [im_array.shape[0], im_array.shape[1]]
            mesh_max = np.max(im_array)
            
            if mesh_max == 0:
                raise ValueError("Image contains no depth information")
            
            # Scale mesh based on depth information
            if len(im_array.shape) == 3:
                scaled_mesh = mesh_size[0] * depth_div_width * im_array[:, :, 0] / mesh_max
            else:
                scaled_mesh = mesh_size[0] * depth_div_width * im_array / mesh_max
            
            # Generate output filenames
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            obj_filename = f"{base_name}_3d.obj"
            mtl_filename = f"{base_name}_3d.mtl"
            obj_path = os.path.join("static/models", obj_filename)
            mtl_path = os.path.join("static/models", mtl_filename)
            
            # Prepare vertices and faces
            vertices = []
            vertex_colors = []
            faces = []
            
            # Generate vertices with colors
            for i in range(mesh_size[0]):
                for j in range(mesh_size[1]):
                    i_scaled = i * aspect_ratio
                    z = scaled_mesh[i, j]
                    vertices.append([i_scaled, j, z])
                    
                    # Calculate vertex color
                    if use_depth_coloring:
                        height_ratio = z / (mesh_size[0] * depth_div_width) if depth_div_width > 0 else 0
                        
                        if color_map == "height":
                            # Height-based gradient (blue -> green -> red)
                            if height_ratio < 0.5:
                                r = 0.0
                                g = height_ratio * 2
                                b = 1.0 - height_ratio * 2
                            else:
                                r = (height_ratio - 0.5) * 2
                                g = 1.0 - (height_ratio - 0.5) * 2
                                b = 0.0
                            vertex_colors.append([r, g, b])
                        elif color_map == "grayscale":
                            # Grayscale based on height
                            gray = 0.3 + height_ratio * 0.7  # 0.3 to 1.0 range
                            vertex_colors.append([gray, gray, gray])
                        else:
                            # Use base color with brightness variation
                            brightness = 0.5 + height_ratio * 0.5
                            vertex_colors.append([
                                base_color[0] * brightness,
                                base_color[1] * brightness,
                                base_color[2] * brightness
                            ])
                    else:
                        # Use uniform base color
                        vertex_colors.append(list(base_color))
            
            # Generate faces (triangles)
            for i in range(mesh_size[0] - 1):
                for j in range(mesh_size[1] - 1):
                    # Vertex indices (OBJ uses 1-based indexing)
                    v1 = i * mesh_size[1] + j + 1
                    v2 = i * mesh_size[1] + (j + 1) + 1
                    v3 = (i + 1) * mesh_size[1] + j + 1
                    v4 = (i + 1) * mesh_size[1] + (j + 1) + 1
                    
                    # Two triangles per quad
                    faces.append([v1, v2, v3])
                    faces.append([v4, v3, v2])
            
            # Write OBJ file
            with open(obj_path, 'w') as f:
                f.write(f"# OBJ file generated from {image_path}\n")
                f.write(f"mtllib {mtl_filename}\n\n")
                
                # Write vertices with colors
                for i, (v, c) in enumerate(zip(vertices, vertex_colors)):
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
                
                f.write(f"\n# {len(faces)} faces\n")
                f.write("usemtl material0\n")
                
                # Write faces
                for face in faces:
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")
            
            # Write MTL file (material library)
            with open(mtl_path, 'w') as f:
                f.write(f"# MTL file for {obj_filename}\n")
                f.write("newmtl material0\n")
                f.write(f"Ka {base_color[0]:.6f} {base_color[1]:.6f} {base_color[2]:.6f}\n")  # Ambient
                f.write(f"Kd {base_color[0]:.6f} {base_color[1]:.6f} {base_color[2]:.6f}\n")  # Diffuse
                f.write("Ks 0.5 0.5 0.5\n")  # Specular
                f.write("Ns 32.0\n")  # Shininess
                f.write("d 1.0\n")  # Transparency (1.0 = opaque)
                f.write("illum 2\n")  # Illumination model
            
            print(f"OBJ model generated: {obj_path}")
            print(f"MTL material generated: {mtl_path}")
            return obj_path, mtl_path
        
    def apply_regional_colors(
            self,
            image_path: str,
            depth_div_width: float,
            color_regions: list[dict],
            aspect_ratio: float = 0.2
        ) -> tuple[str, str]:
            """
            Generate OBJ model with different colors for different regions.
            
            Args:
                image_path: Path to the depth map image
                depth_div_width: Ratio of desired depth to width
                color_regions: List of region definitions, each containing:
                    - 'area': 'center', 'edges', 'top', 'bottom', 'left', 'right', 'all'
                    - 'color': RGB tuple (0-1), e.g., (1.0, 0.0, 0.0) for red
                    - 'threshold': Optional height threshold (0-1)
                aspect_ratio: Height to width ratio
                
            Returns:
                Tuple of (obj_path, mtl_path)
                
            Example:
                color_regions = [
                    {'area': 'center', 'color': (1.0, 0.0, 0.0)},  # Red center
                    {'area': 'edges', 'color': (0.0, 0.0, 1.0)},   # Blue edges
                    {'area': 'all', 'color': (0.8, 0.8, 0.8), 'threshold': 0.5}  # Gray for low areas
                ]
            """
            # Read the image
            im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if im is None:
                raise ValueError(f"Failed to read image from {image_path}")
            
            im_array = np.array(im)
            im_array = 255 - im_array
            im_array = np.rot90(im_array, -1, (0, 1))
            
            mesh_size = [im_array.shape[0], im_array.shape[1]]
            mesh_max = np.max(im_array)
            
            if mesh_max == 0:
                raise ValueError("Image contains no depth information")
            
            if len(im_array.shape) == 3:
                scaled_mesh = mesh_size[0] * depth_div_width * im_array[:, :, 0] / mesh_max
            else:
                scaled_mesh = mesh_size[0] * depth_div_width * im_array / mesh_max
            
            # Default color
            default_color = [0.8, 0.8, 0.8]
            
            # Generate vertices with regional colors
            vertices = []
            vertex_colors = []
            
            H, W = mesh_size[0], mesh_size[1]
            center_h = H // 2
            center_w = W // 2
            edge_threshold = 0.1  # 10% from edges
            
            for i in range(H):
                for j in range(W):
                    i_scaled = i * aspect_ratio
                    z = scaled_mesh[i, j]
                    vertices.append([i_scaled, j, z])
                    
                    # Determine color based on position and regions
                    color = list(default_color)
                    height_ratio = z / (mesh_size[0] * depth_div_width) if depth_div_width > 0 else 0
                    
                    for region in color_regions:
                        area = region.get('area', 'all')
                        region_color = region.get('color', default_color)
                        threshold = region.get('threshold', None)
                        
                        # Check if vertex is in this region
                        in_region = False
                        
                        if area == 'all':
                            in_region = True
                        elif area == 'center':
                            # Center 50% of the model
                            if (abs(i - center_h) < H * 0.25 and abs(j - center_w) < W * 0.25):
                                in_region = True
                        elif area == 'edges':
                            # Outer 10% edges
                            if (i < H * edge_threshold or i > H * (1 - edge_threshold) or
                                j < W * edge_threshold or j > W * (1 - edge_threshold)):
                                in_region = True
                        elif area == 'top':
                            if i < H * 0.3:
                                in_region = True
                        elif area == 'bottom':
                            if i > H * 0.7:
                                in_region = True
                        elif area == 'left':
                            if j < W * 0.3:
                                in_region = True
                        elif area == 'right':
                            if j > W * 0.7:
                                in_region = True
                        
                        # Apply threshold if specified
                        if threshold is not None:
                            in_region = in_region and (height_ratio >= threshold)
                        
                        if in_region:
                            color = list(region_color)
                            break  # Use first matching region
                    
                    vertex_colors.append(color)
            
            # Generate faces
            faces = []
            for i in range(H - 1):
                for j in range(W - 1):
                    v1 = i * W + j + 1
                    v2 = i * W + (j + 1) + 1
                    v3 = (i + 1) * W + j + 1
                    v4 = (i + 1) * W + (j + 1) + 1
                    
                    faces.append([v1, v2, v3])
                    faces.append([v4, v3, v2])
            
            # Write files
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            obj_filename = f"{base_name}_colored_3d.obj"
            mtl_filename = f"{base_name}_colored_3d.mtl"
            obj_path = os.path.join("static/models", obj_filename)
            mtl_path = os.path.join("static/models", mtl_filename)
            
            with open(obj_path, 'w') as f:
                f.write(f"# OBJ file with regional colors\n")
                f.write(f"mtllib {mtl_filename}\n\n")
                
                for v, c in zip(vertices, vertex_colors):
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
                
                f.write(f"\n# {len(faces)} faces\n")
                f.write("usemtl material0\n")
                
                for face in faces:
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")
            
            with open(mtl_path, 'w') as f:
                f.write("# MTL file for regional colored model\n")
                f.write("newmtl material0\n")
                f.write("Ka 0.8 0.8 0.8\n")
                f.write("Kd 0.8 0.8 0.8\n")
                f.write("Ks 0.5 0.5 0.5\n")
                f.write("Ns 32.0\n")
                f.write("d 1.0\n")
                f.write("illum 2\n")
            
            print(f"Colored OBJ model generated: {obj_path}")
            return obj_path, mtl_path
        
        
        # def convert_depth_to_stl(
        #     self,
        #     image_path: str,
        #     depth_div_width: float,
        #     aspect_ratio: float = 1.0,
        #     reduce_factor: float = 0.5,
        #     smooth: bool = True
        # ) -> str:
        #     """
        #     Convert a depth map image to a solid STL 3D model.

        #     Args:
        #         image_path: Path to the depth map image
        #         depth_div_width: Ratio of desired depth to width
        #         aspect_ratio: height scaling
        #         reduce_factor: resolution scaling factor (<1 reduces file size)
        #         smooth: apply gaussian blur to reduce noise

        #     Returns:
        #         Path to generated STL file
        #     """
        #     import os
        #     import cv2
        #     import numpy as np
        #     from stl import mesh

        #     # ------------ 1. Load image ------------
        #     im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #     if im is None:
        #         raise ValueError(f"Failed to read image from {image_path}")

        #     # Reduce image resolution if needed
        #     if reduce_factor < 1.0:
        #         new_w = int(im.shape[1] * reduce_factor)
        #         new_h = int(im.shape[0] * reduce_factor)
        #         im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

        #     # Smooth to reduce noise
        #     if smooth:
        #         im = cv2.GaussianBlur(im, (7, 7), 0)

        #     # Convert to array
        #     im_array = 255 - np.array(im)
        #     im_array = np.rot90(im_array, -1, (0, 1))

        #     mesh_size = [im_array.shape[0], im_array.shape[1]]
        #     mesh_max = np.max(im_array)
        #     if mesh_max == 0:
        #         raise ValueError("Image contains no depth information")

        #     # ------------ 2. Build height map ------------
        #     if len(im_array.shape) == 3:
        #         height_map = mesh_size[0] * depth_div_width * im_array[:, :, 0] / mesh_max
        #     else:
        #         height_map = mesh_size[0] * depth_div_width * im_array / mesh_max

        #     H, W = mesh_size

        #     # Number of triangles for the top surface
        #     top_tri_count = (H - 1) * (W - 1) * 2

        #     # ------------ 3. Build TOP surface ------------
        #     top_mesh = mesh.Mesh(np.zeros(top_tri_count, dtype=mesh.Mesh.dtype))

        #     idx = 0
        #     for i in range(H - 1):
        #         for j in range(W - 1):

        #             i0 = i * aspect_ratio
        #             i1 = (i + 1) * aspect_ratio

        #             # Triangle 1
        #             top_mesh.vectors[idx][0] = [i0, j, height_map[i, j]]
        #             top_mesh.vectors[idx][1] = [i0, j+1, height_map[i, j+1]]
        #             top_mesh.vectors[idx][2] = [i1, j, height_map[i+1, j]]
        #             idx += 1

        #             # Triangle 2
        #             top_mesh.vectors[idx][0] = [i1, j+1, height_map[i+1, j+1]]
        #             top_mesh.vectors[idx][1] = [i0, j+1, height_map[i, j+1]]
        #             top_mesh.vectors[idx][2] = [i1, j, height_map[i+1, j]]
        #             idx += 1

        #     # ------------ 4. Bottom surface (z = 0) ------------
        #     bottom_mesh = mesh.Mesh(np.zeros(top_tri_count, dtype=mesh.Mesh.dtype))

        #     idx = 0
        #     for i in range(H - 1):
        #         for j in range(W - 1):

        #             i0 = i * aspect_ratio
        #             i1 = (i + 1) * aspect_ratio

        #             # Triangle 1
        #             bottom_mesh.vectors[idx][0] = [i0, j, 0]
        #             bottom_mesh.vectors[idx][1] = [i1, j, 0]
        #             bottom_mesh.vectors[idx][2] = [i0, j+1, 0]
        #             idx += 1

        #             # Triangle 2
        #             bottom_mesh.vectors[idx][0] = [i1, j+1, 0]
        #             bottom_mesh.vectors[idx][1] = [i0, j+1, 0]
        #             bottom_mesh.vectors[idx][2] = [i1, j, 0]
        #             idx += 1

        #     # ------------ 5. Build SIDE walls ------------
        #     side_meshes = []

        #     # Helper to add a wall strip
        #     def add_wall(x1, y1, z1, x2, y2, z2):
        #         """Two points top; bottom is z=0."""
        #         m = mesh.Mesh(np.zeros(2, dtype=mesh.Mesh.dtype))

        #         # Triangle 1
        #         m.vectors[0][0] = [x1, y1, z1]
        #         m.vectors[0][1] = [x1, y1, 0]
        #         m.vectors[0][2] = [x2, y2, z2]

        #         # Triangle 2
        #         m.vectors[1][0] = [x2, y2, z2]
        #         m.vectors[1][1] = [x1, y1, 0]
        #         m.vectors[1][2] = [x2, y2, 0]

        #         return m

        #     # Front wall j = 0
        #     for i in range(H - 1):
        #         z1 = height_map[i, 0]
        #         z2 = height_map[i+1, 0]
        #         side_meshes.append(add_wall(i*aspect_ratio, 0, z1, (i+1)*aspect_ratio, 0, z2))

        #     # Back wall j = W - 1
        #     for i in range(H - 1):
        #         z1 = height_map[i, W-1]
        #         z2 = height_map[i+1, W-1]
        #         side_meshes.append(add_wall(i*aspect_ratio, W-1, z1, (i+1)*aspect_ratio, W-1, z2))

        #     # Left wall i = 0
        #     for j in range(W - 1):
        #         z1 = height_map[0, j]
        #         z2 = height_map[0, j+1]
        #         side_meshes.append(add_wall(0, j, z1, 0, j+1, z2))

        #     # Right wall i = H - 1
        #     for j in range(W - 1):
        #         z1 = height_map[H-1, j]
        #         z2 = height_map[H-1, j+1]
        #         side_meshes.append(add_wall((H-1)*aspect_ratio, j, z1, (H-1)*aspect_ratio, j+1, z2))

        #     # ------------ 6. Combine all meshes ------------
        #     all_meshes = [top_mesh, bottom_mesh] + side_meshes

        #     # Merge into single numpy array
        #     total_triangles = sum(m.vectors.shape[0] for m in all_meshes)
        #     full_mesh = mesh.Mesh(np.zeros(total_triangles, dtype=mesh.Mesh.dtype))

        #     idx = 0
        #     for m in all_meshes:
        #         tri = m.vectors.shape[0]
        #         full_mesh.vectors[idx:idx+tri] = m.vectors
        #         idx += tri

        #     # ------------ 7. Save STL ------------
        #     base_name = os.path.splitext(os.path.basename(image_path))[0]
        #     output_filename = f"{base_name}_solid.stl"
        #     output_path = os.path.join("static/models", output_filename)
        #     full_mesh.save(output_path)

        #     return output_path


class DepthTo3DService:
    """Service to convert depth maps to 3D STL models"""
    
    @staticmethod
    def convert_depth_to_stl(
        image_data: bytes,
        depth_div_width: float,
        output_path: str,
        aspect_ratio: float = 1.0
    ) -> str:
        """
        Convert a depth map image to an STL 3D model
        
        Args:
            image_data: Raw image bytes
            depth_div_width: Ratio of desired depth to width (e.g., 0.5)
            output_path: Path where to save the STL file
            aspect_ratio: Height to width ratio for the model (default: 1.0)
            
        Returns:
            Path to the generated STL file
        """
        # Decode image from bytes
        nparr = np.frombuffer(image_data, np.uint8)
        im = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if im is None:
            raise ValueError("Failed to decode image")
        
        # Process image array
        im_array = np.array(im)
        im_array = np.rot90(im_array, -1, (0, 1))
        
        mesh_size = [im_array.shape[0], im_array.shape[1]]
        mesh_max = np.max(im_array)
        
        if mesh_max == 0:
            raise ValueError("Image contains no depth information (all pixels are black)")
        
        # Scale mesh based on depth information
        if len(im_array.shape) == 3:
            # Color image - use first channel
            scaled_mesh = mesh_size[0] * depth_div_width * im_array[:, :, 0] / mesh_max
        else:
            # Grayscale image
            scaled_mesh = mesh_size[0] * depth_div_width * im_array / mesh_max
        
        # Create mesh
        mesh_shape = mesh.Mesh(
            np.zeros((mesh_size[0] - 1) * (mesh_size[1] - 1) * 2, dtype=mesh.Mesh.dtype)
        )
        
        # Generate triangles for the mesh
        for i in range(0, mesh_size[0] - 1):
            for j in range(0, mesh_size[1] - 1):
                mesh_num = i * (mesh_size[1] - 1) + j
                
                # Apply aspect ratio to i coordinate (height)
                i_scaled = i * aspect_ratio
                i1_scaled = (i + 1) * aspect_ratio
                
                # First triangle
                mesh_shape.vectors[2 * mesh_num][2] = [i_scaled, j, scaled_mesh[i, j]]
                mesh_shape.vectors[2 * mesh_num][1] = [i_scaled, j + 1, scaled_mesh[i, j + 1]]
                mesh_shape.vectors[2 * mesh_num][0] = [i1_scaled, j, scaled_mesh[i + 1, j]]
                
                # Second triangle
                mesh_shape.vectors[2 * mesh_num + 1][0] = [i1_scaled, j + 1, scaled_mesh[i + 1, j + 1]]
                mesh_shape.vectors[2 * mesh_num + 1][1] = [i_scaled, j + 1, scaled_mesh[i, j + 1]]
                mesh_shape.vectors[2 * mesh_num + 1][2] = [i1_scaled, j, scaled_mesh[i + 1, j]]
        
        # Save mesh to file
        mesh_shape.save(output_path)
        return output_path
