# Prompt2Printable 

An AI-powered web application that converts keychain image into a 3D-printable model.

## Tech Stack

### Backend
- **FastAPI** - High-performance web framework
- **OpenAI GPT-4o** - Image understanding and generation
- **OpenCV** - Image processing
- **NumPy** - Numerical computing
- **numpy-stl** - 3D model generation

### Frontend
- **Vanilla JavaScript** - No framework dependencies
- **HTML5 Canvas** - Silhouette editing functionality
- **CSS3** - Responsive UI design

## Quick Start

### Installation

1. **Clone the repository**
2. **Create virtual environment**
```bash
python3 -m venv env
source env/bin/activate  # macOS/Linux
# or env\Scripts\activate  # Windows
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```
4. **Configure API Key**
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```
5. **Start the server**
```bash
python main.py
```
6. **Access the application**

Open your browser and navigate to: `http://localhost:8000`

## User Guide

### Basic Workflow

1. **Upload Image**
2. **Generate Model**
3. **Edit Silhouette** (Optional)
   - Click "Edit Silhouette" to open the editor
   - Use pen tool to add white areas
   - Use eraser tool to remove areas
   - Enter editing instructions for AI optimization
4. **Refine Model** (Optional)
   Enter natural language instructions in the chat, for example:
   ```
   Make it 20% bigger
   Add texture to the surface
   Increase the depth
   Smooth out the edges
   ```
5. **Adjust Colors** (Optional)
   
   Describe desired colors in natural language:
   ```
   Make it redder
   Add more pink tone
   Center should be yellow, edges blue
   Not blue enough
   ```

6. **Download Models**
   - **STL Format**: For 3D printing (no color)
   - **OBJ Format**: With materials and colors (download both OBJ and MTL files)

## Project Structure

```
prompt2printable/
├── main.py              # FastAPI main application
├── services.py          # Core business logic
├── prompts.py           # AI prompt templates
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create manually)
├── static/              # Frontend resources
│   ├── index.html       # Main page
│   ├── app.js           # Frontend logic
│   ├── styles.css       # Stylesheet
│   ├── uploads/         # Uploaded images
│   ├── processed/       # Processed silhouettes
│   └── models/          # Generated 3D models
└── env/                 # Python virtual environment
```

## API Endpoints

Main endpoints:

- `POST /upload` - Upload image
- `POST /analyze` - Analyze image proportions
- `POST /generate-silhouette` - Generate silhouette
- `POST /edit-silhouette` - Edit silhouette
- `POST /convert-to-3d` - Convert to STL model
- `POST /convert-to-obj` - Convert to OBJ model (with color)
- `POST /refine-model` - Refine model
- `POST /adjust-colors` - Adjust colors
- `GET /download-model/{filename}` - Download model

Full API documentation: Visit `http://localhost:8000/docs` after starting the server

