import os
from dotenv import load_dotenv
load_dotenv()

# 检查并设置 OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please create a .env file with your API key."
    )
os.environ["OPENAI_API_KEY"] = api_key

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from services import ImageService


app = FastAPI(title="Keychain Generator API")
service = ImageService()

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该设置具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisResponse(BaseModel):
    success: bool
    ratio_string: Optional[str] = None
    data: Optional[dict] = None
    error: Optional[str] = None

class ImageResponse(BaseModel):
    success: bool
    image_url: str
    local_path: str

class ModelResponse(BaseModel):
    success: bool
    model_url: str
    local_path: str

class RefinementResponse(BaseModel):
    success: bool
    refined_image_url: str
    refined_image_path: str
    new_model_url: Optional[str] = None
    new_model_path: Optional[str] = None

class OBJModelResponse(BaseModel):
    success: bool
    obj_url: str
    obj_path: str
    mtl_url: Optional[str] = None
    mtl_path: Optional[str] = None

# 根路由 - 提供前端页面（必须在 static mount 之前）
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """提供前端页面"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)

# Mount static directory - 必须在路由定义之后
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/upload", response_model=ImageResponse)
async def upload_image(file: UploadFile = File(...)):
    """Uploads an image and converts it to PNG."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    try:
        contents = await file.read()
        saved_path = service.convert_to_png(contents, file.filename)
        
        # Return a URL that can be accessed via the browser
        url_path = f"/static/uploads/{os.path.basename(saved_path)}"
        return {"success": True, "image_url": url_path, "local_path": saved_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(local_path: str = Form(...)):
    """Analyzes the uploaded image for proportions."""
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Image file not found")
        
    try:
        result = service.analyze_proportions(local_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-silhouette", response_model=ImageResponse)
async def generate_silhouette(local_path: str = Form(...)):
    """Generates a black and white silhouette from the image."""
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Image file not found")
        
    try:
        print(f"Generating silhouette for {local_path}")
        output_path = service.generate_silhouette(local_path)

        print(f"Silhouette saved to {output_path}")
        url_path = f"/static/processed/{os.path.basename(output_path)}"
        print(f"Accessible at {url_path}")
        return {"success": True, "image_url": url_path, "local_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/edit-silhouette", response_model=ImageResponse)
async def edit_silhouette(
    local_path: str = Form(...), 
    instructions: str = Form("")
):
    """
    Uploads an image with red marks and applies edits based on instructions.
    The file uploaded here should be the silhouette modified by the user.
    """
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    try:
        output_path = service.edit_silhouette(local_path, instructions=instructions)
        
        url_path = f"/static/processed/{os.path.basename(output_path)}"
        return {"success": True, "image_url": url_path, "local_path": output_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/convert-to-3d", response_model=ModelResponse)
async def convert_to_3d(
    local_path: str = Form(...),
    depth_div_width: float = Form(..., description="Ratio of desired depth to width (e.g., 0.5)", gt=0),
    aspect_ratio: float = Form(1.0, description="Height to width ratio (default: 1.0)", gt=0)
):
    """
    Converts a depth map image to a 3D STL model.
    
    Parameters:
    - local_path: Path to the depth map image (usually from generate-silhouette or edit-silhouette)
    - depth_div_width: Ratio of depth to width (e.g., 0.5 for 10cm deep and 20cm wide)
    - aspect_ratio: Height to width ratio (1.0 = original proportions, >1.0 = taller, <1.0 = wider)
    """
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        # Convert the depth map to 3D model
        output_path = service.convert_depth_to_stl(
            local_path, 
            depth_div_width, 
            aspect_ratio
        )
        
        # Return URL to download the STL file
        url_path = f"/static/models/{os.path.basename(output_path)}"
        return {"success": True, "model_url": url_path, "local_path": output_path}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refine-model", response_model=ImageResponse)
async def refine_model(
    depth_map_path: str = Form(..., description="Path to existing depth map/silhouette"),
    refinement_instructions: str = Form(..., description="Description of desired changes")
):
    """
    Refines an existing depth map based on user instructions.
    
    This endpoint allows users to make adjustments to size, texture, height, details, etc.
    without regenerating the entire model from scratch.
    
    Parameters:
    - depth_map_path: Path to the existing depth map (black/white/grayscale image)
    - refinement_instructions: Natural language description of changes
      Examples:
      - "Make it 20% bigger"
      - "Add texture to the surface"
      - "Increase the depth by 50%"
      - "Smooth out the edges"
      - "Add embossed text saying 'CUSTOM'"
    
    Returns:
    - Refined depth map image (can be used with /convert-to-3d endpoint)
    """
    if not os.path.exists(depth_map_path):
        raise HTTPException(status_code=404, detail="Depth map file not found")
    
    if not refinement_instructions or refinement_instructions.strip() == "":
        raise HTTPException(status_code=400, detail="Refinement instructions are required")
    
    try:
        refined_path = service.refine_model(depth_map_path, refinement_instructions)
        url_path = f"/static/processed/{os.path.basename(refined_path)}"
        return {"success": True, "image_url": url_path, "local_path": refined_path}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refine-and-regenerate", response_model=RefinementResponse)
async def refine_and_regenerate(
    depth_map_path: str = Form(..., description="Path to existing depth map"),
    refinement_instructions: str = Form(..., description="Description of desired changes"),
    depth_div_width: float = Form(..., description="Ratio of depth to width", gt=0),
    aspect_ratio: float = Form(1.0, description="Height to width ratio", gt=0)
):
    """
    Complete workflow: refines depth map AND generates new 3D model in one step.
    
    This is a convenience endpoint that combines /refine-model and /convert-to-3d.
    
    Parameters:
    - depth_map_path: Path to existing depth map
    - refinement_instructions: Natural language description of changes
    - depth_div_width: Ratio of depth to width for 3D model
    - aspect_ratio: Height to width ratio for 3D model
    
    Returns:
    - Both the refined depth map and the new STL model file
    """
    if not os.path.exists(depth_map_path):
        raise HTTPException(status_code=404, detail="Depth map file not found")
    
    if not refinement_instructions or refinement_instructions.strip() == "":
        raise HTTPException(status_code=400, detail="Refinement instructions are required")
    
    try:
        refined_depth_map, new_stl_path = service.refine_and_regenerate_3d(
            depth_map_path,
            refinement_instructions,
            depth_div_width,
            aspect_ratio
        )
        
        refined_url = f"/static/processed/{os.path.basename(refined_depth_map)}"
        model_url = f"/static/models/{os.path.basename(new_stl_path)}"
        
        return {
            "success": True,
            "refined_image_url": refined_url,
            "refined_image_path": refined_depth_map,
            "new_model_url": model_url,
            "new_model_path": new_stl_path
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/convert-to-obj", response_model=OBJModelResponse)
async def convert_to_obj(
    local_path: str = Form(..., description="Path to the depth map image"),
    depth_div_width: float = Form(..., description="Ratio of depth to width", gt=0),
    aspect_ratio: float = Form(1.0, description="Height to width ratio", gt=0),
    base_color_r: float = Form(0.8, description="Red component (0-1)", ge=0, le=1),
    base_color_g: float = Form(0.8, description="Green component (0-1)", ge=0, le=1),
    base_color_b: float = Form(0.8, description="Blue component (0-1)", ge=0, le=1),
    use_depth_coloring: bool = Form(False, description="Apply color based on height"),
    color_map: str = Form("grayscale", description="Color mapping: grayscale, height, or custom")
):
    """
    Converts a depth map to a colored OBJ 3D model.
    
    Parameters:
    - local_path: Path to depth map image
    - depth_div_width: Ratio of depth to width
    - aspect_ratio: Height to width ratio
    - base_color_r/g/b: Base color RGB values (0-1)
    - use_depth_coloring: Enable height-based coloring
    - color_map: Color mapping mode
      - "grayscale": Brightness based on height
      - "height": Blue(low) -> Green(mid) -> Red(high)
      - "custom": Base color with brightness variation
    
    Returns:
    - OBJ file and MTL material file URLs
    """
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        base_color = (base_color_r, base_color_g, base_color_b)
        
        obj_path, mtl_path = service.convert_depth_to_obj(
            local_path,
            depth_div_width,
            aspect_ratio,
            base_color,
            use_depth_coloring,
            color_map
        )
        
        obj_url = f"/static/models/{os.path.basename(obj_path)}"
        mtl_url = f"/static/models/{os.path.basename(mtl_path)}"
        
        return {
            "success": True,
            "obj_url": obj_url,
            "obj_path": obj_path,
            "mtl_url": mtl_url,
            "mtl_path": mtl_path
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/convert-to-obj-regional", response_model=OBJModelResponse)
async def convert_to_obj_regional(
    local_path: str = Form(..., description="Path to the depth map image"),
    depth_div_width: float = Form(..., description="Ratio of depth to width", gt=0),
    aspect_ratio: float = Form(1.0, description="Height to width ratio", gt=0),
    color_config: str = Form(..., description="JSON string of color regions configuration")
):
    """
    Converts a depth map to OBJ with different colors for different regions.
    
    Parameters:
    - local_path: Path to depth map
    - depth_div_width: Depth to width ratio
    - aspect_ratio: Height to width ratio
    - color_config: JSON array of region configurations
    
    Example color_config:
    ```json
    [
        {"area": "center", "color": [1.0, 0.0, 0.0]},
        {"area": "edges", "color": [0.0, 0.0, 1.0]},
        {"area": "top", "color": [0.0, 1.0, 0.0], "threshold": 0.5}
    ]
    ```
    
    Supported areas: "all", "center", "edges", "top", "bottom", "left", "right"
    Optional threshold: Height threshold (0-1) to filter vertices
    """
    if not os.path.exists(local_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    try:
        import json
        color_regions = json.loads(color_config)
        
        # Convert color arrays to tuples
        for region in color_regions:
            if 'color' in region and isinstance(region['color'], list):
                region['color'] = tuple(region['color'])
        
        obj_path, mtl_path = service.apply_regional_colors(
            local_path,
            depth_div_width,
            color_regions,
            aspect_ratio
        )
        
        obj_url = f"/static/models/{os.path.basename(obj_path)}"
        mtl_url = f"/static/models/{os.path.basename(mtl_path)}"
        
        return {
            "success": True,
            "obj_url": obj_url,
            "obj_path": obj_path,
            "mtl_url": mtl_url,
            "mtl_path": mtl_path
        }
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in color_config: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/adjust-colors")
async def adjust_colors(
    user_instruction: str = Form(..., description="Natural language color adjustment request"),
    current_config: str = Form(..., description="Current color configuration as JSON string")
):
    """
    Adjusts model colors based on natural language instructions.
    
    This endpoint uses GPT-4o to interpret casual color descriptions like:
    - "make it redder"
    - "add more pink tone"
    - "not blue enough"
    - "center should be more yellow, edges darker"
    
    Parameters:
    - user_instruction: Natural language description of desired color changes
    - current_config: Current color configuration as JSON string
    
    Current config format for single color:
    ```json
    {
        "mode": "single",
        "base_color": [0.8, 0.8, 0.8]
    }
    ```
    
    Current config format for regional colors:
    ```json
    {
        "mode": "regional",
        "color_config": [
            {"area": "center", "color": [1.0, 0.0, 0.0]},
            {"area": "edges", "color": [0.0, 0.0, 1.0]}
        ]
    }
    ```
    
    Returns:
    Updated color configuration with the same structure plus an "explanation" field
    """
    try:
        import json
        
        # Parse the current configuration
        try:
            current_config_dict = json.loads(current_config)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=400, 
                detail="Invalid JSON in current_config parameter"
            )
        
        # Validate current_config structure
        if "mode" not in current_config_dict:
            raise HTTPException(
                status_code=400,
                detail="current_config must contain 'mode' field ('single' or 'regional')"
            )
        
        # Call the color adjustment service
        adjusted_config = service.adjust_model_colors(
            user_instruction=user_instruction,
            current_config=current_config_dict
        )
        
        return {
            "success": True,
            "updated_config": adjusted_config,
            "explanation": adjusted_config.get("explanation", "Color adjustment completed"),
            "original_instruction": user_instruction
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Color adjustment failed: {str(e)}")

@app.get("/download-model/{filename}")
async def download_model(filename: str):
    """Downloads the generated STL, OBJ, or MTL model file."""
    file_path = os.path.join("static", "models", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # Determine media type based on file extension
    if filename.endswith('.stl'):
        media_type = 'application/vnd.ms-pki.stl'
    elif filename.endswith('.obj'):
        media_type = 'model/obj'
    elif filename.endswith('.mtl'):
        media_type = 'model/mtl'
    else:
        media_type = 'application/octet-stream'
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)