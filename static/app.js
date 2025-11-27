// API 基础 URL
const API_BASE = 'http://localhost:8000';

// 状态管理
let state = {
    uploadedImagePath: null,
    currentDepthMapPath: null,
    currentStlPath: null,
    currentObjPath: null,
    currentMtlPath: null,
    currentColorConfig: {
        mode: 'single',
        base_color: [0.8, 0.8, 0.8]
    }
};

// 编辑器状态
let editorState = {
    canvas: null,
    ctx: null,
    isDrawing: false,
    currentTool:"pen",
    brushSize:15,
    history: [],
    originalImageSrc: null,
    lastX: 0,
    lastY: 0
};

// DOM 元素
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const previewImg = document.getElementById('previewImg');
const imagePreview = document.getElementById('imagePreview');
const promptInput = document.getElementById('promptInput');
const generateBtn = document.getElementById('generateBtn');
const modelPreview = document.getElementById('modelPreview');
const cadPreview = document.getElementById('cadPreview');
const downloadStl = document.getElementById('downloadStl');
const downloadObj = document.getElementById('downloadObj');
const refinementInput = document.getElementById('refinementInput');
const refineBtn = document.getElementById('refineBtn');
const chatContainer = document.getElementById('chatContainer');
const loadingOverlay = document.getElementById('loadingOverlay');

// 文件上传处理
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    fileName.textContent = file.name;
    
    // 显示本地预览
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewImg.style.display = 'block';
        imagePreview.querySelector('.no-image-text').style.display = 'none';
    };
    reader.readAsDataURL(file);

    // 上传到服务器
    await uploadImage(file);
});

// 上传图片到服务器
async function uploadImage(file) {
    showLoading('Uploading image...');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (data.success) {
            state.uploadedImagePath = data.local_path;
            addChatMessage('Image uploaded successfully!', 'ai');
            console.log('Uploaded path:', data.local_path);
        } else {
            throw new Error('Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        addChatMessage('Failed to upload image. Please try again.', 'ai');
    } finally {
        hideLoading();
    }
}

// 生成模型主流程
generateBtn.addEventListener('click', async () => {
    const prompt = promptInput.value.trim();
    
    if (!state.uploadedImagePath && !prompt) {
        addChatMessage('Please upload an image or enter a prompt.', 'ai');
        return;
    }

    generateBtn.disabled = true;
    
    try {
        // 步骤1: 分析图片比例（如果有上传图片）
        if (state.uploadedImagePath) {
            await analyzeImage();
        }

        // 步骤2: 生成轮廓
        await generateSilhouette();

        // 步骤3: 生成3D模型（STL和OBJ）
        await generate3DModels();

        addChatMessage('Model generated successfully! You can download it now.', 'ai');
    } catch (error) {
        console.error('Generation error:', error);
        addChatMessage('Failed to generate model. Please try again.', 'ai');
    } finally {
        generateBtn.disabled = false;
    }
});

// 分析图片
async function analyzeImage() {
    showLoading('Analyzing image proportions...');
    
    const formData = new FormData();
    formData.append('local_path', state.uploadedImagePath);

    const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    
    if (data.success) {
        addChatMessage(`Analysis complete: ${data.ratio_string}`, 'ai');
        console.log('Analysis data:', data.data);
    }
    
    hideLoading();
}

// 生成轮廓
// async function generateSilhouette() {
//     showLoading('Generating silhouette...');
    
//     const formData = new FormData();
//     formData.append('local_path', state.uploadedImagePath);

//     const response = await fetch(`${API_BASE}/generate-silhouette`, {
//         method: 'POST',
//         body: formData
//     });

//     const data = await response.json();
    
//     if (data.success) {
//         state.currentDepthMapPath = data.local_path;
        
//         // 显示轮廓预览
//         modelPreview.src = `${API_BASE}${data.image_url}`;
//         modelPreview.style.display = 'block';
//         cadPreview.querySelector('.preview-placeholder').style.display = 'none';
        
//         addChatMessage('Silhouette generated!', 'ai');
//     }
    
//     hideLoading();
// }
async function generateSilhouette() {
    showLoading('Generating silhouette...');
    
    const formData = new FormData();
    formData.append('local_path', state.uploadedImagePath);

    try {
        const response = await fetch(`${API_BASE}/generate-silhouette`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (data.success) {
            state.currentDepthMapPath = data.local_path;
            
            // 显示轮廓预览
            modelPreview.src = `${API_BASE}${data.image_url}`;
            modelPreview.style.display = 'block';
            cadPreview.querySelector('.preview-placeholder').style.display = 'none';
            
            addChatMessage('✅ Silhouette generated successfully!', 'ai');
            
            // 创建并添加编辑按钮
            const editMessageDiv = document.createElement('div');
            editMessageDiv.className = 'chat-message ai-message';
            editMessageDiv.innerHTML = `
                <p>You can manually edit the silhouette to perfect it:</p>
                <button onclick="openSilhouetteEditor()" 
                        style="margin-top: 10px; padding: 10px 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold;">
                    ✏️ Edit Silhouette Manually
                </button>
            `;
            chatContainer.appendChild(editMessageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            
            console.log('Edit button added to chat');
        } else {
            addChatMessage('Failed to generate silhouette.', 'ai');
        }
    } catch (error) {
        console.error('Error generating silhouette:', error);
        addChatMessage('Error generating silhouette: ' + error.message, 'ai');
    } finally {
        hideLoading();
    }
}

// 生成3D模型（STL和OBJ）
async function generate3DModels() {
    showLoading('Converting to 3D models...');
    
    const depth_div_width = 0.2; // 默认深度比例
    const aspect_ratio = 1.0;
    
    // 生成STL
    const stlFormData = new FormData();
    stlFormData.append('local_path', state.currentDepthMapPath);
    stlFormData.append('depth_div_width', depth_div_width);
    stlFormData.append('aspect_ratio', aspect_ratio);

    const stlResponse = await fetch(`${API_BASE}/convert-to-3d`, {
        method: 'POST',
        body: stlFormData
    });

    const stlData = await stlResponse.json();
    
    if (stlData.success) {
        state.currentStlPath = stlData.local_path;
        downloadStl.disabled = false;
    }

    // 生成OBJ（带颜色）
    const objFormData = new FormData();
    objFormData.append('local_path', state.currentDepthMapPath);
    objFormData.append('depth_div_width', depth_div_width);
    objFormData.append('aspect_ratio', aspect_ratio);
    objFormData.append('base_color_r', state.currentColorConfig.base_color[0]);
    objFormData.append('base_color_g', state.currentColorConfig.base_color[1]);
    objFormData.append('base_color_b', state.currentColorConfig.base_color[2]);
    objFormData.append('use_depth_coloring', 'false');

    const objResponse = await fetch(`${API_BASE}/convert-to-obj`, {
        method: 'POST',
        body: objFormData
    });

    const objData = await objResponse.json();
    
    if (objData.success) {
        state.currentObjPath = objData.obj_path;
        state.currentMtlPath = objData.mtl_path;
        downloadObj.disabled = false;
    }
    
    hideLoading();
}

// 模型微调
refineBtn.addEventListener('click', async () => {
    const refinementText = refinementInput.value.trim();
    
    if (!refinementText) {
        addChatMessage('Please enter a refinement instruction.', 'ai');
        return;
    }

    if (!state.currentDepthMapPath) {
        addChatMessage('Please generate a model first.', 'ai');
        return;
    }

    addChatMessage(refinementText, 'user');
    refinementInput.value = '';
    refineBtn.disabled = true;

    // 检查是否是颜色调整指令
    const colorKeywords = ['color', 'red', 'blue', 'green', 'pink', 'orange', 'yellow', 'purple', 
                          'darker', 'lighter', 'saturated', '颜色', '红', '蓝', '绿', '粉'];
    const isColorAdjustment = colorKeywords.some(keyword => 
        refinementText.toLowerCase().includes(keyword.toLowerCase())
    );

    try {
        if (isColorAdjustment) {
            await adjustColors(refinementText);
        } else {
            await refineModel(refinementText);
        }
    } catch (error) {
        console.error('Refinement error:', error);
        addChatMessage('Failed to refine model. Please try again.', 'ai');
    } finally {
        refineBtn.disabled = false;
    }
});

// 调整颜色
async function adjustColors(instruction) {
    showLoading('Adjusting colors...');
    
    const formData = new FormData();
    formData.append('user_instruction', instruction);
    formData.append('current_config', JSON.stringify(state.currentColorConfig));

    const response = await fetch(`${API_BASE}/adjust-colors`, {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    
    if (data.success) {
        // 更新颜色配置
        state.currentColorConfig = data.updated_config;
        
        addChatMessage(`Color adjusted: ${data.explanation}`, 'ai');
        
        // 重新生成OBJ模型
        await regenerateObjWithNewColors();
    }
    
    hideLoading();
}

// 使用新颜色重新生成OBJ
async function regenerateObjWithNewColors() {
    showLoading('Regenerating model with new colors...');
    
    const formData = new FormData();
    formData.append('local_path', state.currentDepthMapPath);
    formData.append('depth_div_width', '0.2');
    formData.append('aspect_ratio', '1.0');
    
    if (state.currentColorConfig.mode === 'single') {
        formData.append('base_color_r', state.currentColorConfig.base_color[0]);
        formData.append('base_color_g', state.currentColorConfig.base_color[1]);
        formData.append('base_color_b', state.currentColorConfig.base_color[2]);
        formData.append('use_depth_coloring', 'false');

        const response = await fetch(`${API_BASE}/convert-to-obj`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (data.success) {
            state.currentObjPath = data.obj_path;
            state.currentMtlPath = data.mtl_path;
            addChatMessage('OBJ model updated with new colors!', 'ai');
        }
    } else if (state.currentColorConfig.mode === 'regional') {
        formData.append('color_config', JSON.stringify(state.currentColorConfig.color_config));

        const response = await fetch(`${API_BASE}/convert-to-obj-regional`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (data.success) {
            state.currentObjPath = data.obj_path;
            state.currentMtlPath = data.mtl_path;
            addChatMessage('OBJ model updated with regional colors!', 'ai');
        }
    }
    
    hideLoading();
}

// 微调模型（非颜色调整）
async function refineModel(instruction) {
    showLoading('Refining model...');
    
    const formData = new FormData();
    formData.append('depth_map_path', state.currentDepthMapPath);
    formData.append('refinement_instructions', instruction);
    formData.append('depth_div_width', '0.2');
    formData.append('aspect_ratio', '1.0');

    const response = await fetch(`${API_BASE}/refine-and-regenerate`, {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    
    if (data.success) {
        // 更新深度图路径
        state.currentDepthMapPath = data.refined_image_path;
        state.currentStlPath = data.new_model_path;
        
        // 更新预览
        modelPreview.src = `${API_BASE}${data.refined_image_url}?t=${Date.now()}`;
        
        addChatMessage('Model refined successfully!', 'ai');
        
        // 同时更新OBJ模型
        await regenerateObjWithNewColors();
    }
    
    hideLoading();
}

// 下载模型
downloadStl.addEventListener('click', () => {
    if (state.currentStlPath) {
        const filename = state.currentStlPath.split('/').pop();
        window.open(`${API_BASE}/download-model/${filename}`, '_blank');
    }
});

downloadObj.addEventListener('click', () => {
    if (state.currentObjPath) {
        const objFilename = state.currentObjPath.split('/').pop();
        const mtlFilename = state.currentMtlPath.split('/').pop();
        
        // 下载OBJ文件
        window.open(`${API_BASE}/download-model/${objFilename}`, '_blank');
        
        // 延迟下载MTL文件
        setTimeout(() => {
            window.open(`${API_BASE}/download-model/${mtlFilename}`, '_blank');
        }, 500);
    }
});

// 添加聊天消息
function addChatMessage(text, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}-message`;
    messageDiv.innerHTML = `<p>${text}</p>`;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 显示/隐藏加载状态
function showLoading(text = 'Processing...') {
    loadingOverlay.querySelector('.loading-text').textContent = text;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}


// 打开轮廓编辑器
function openSilhouetteEditor() {
    if (!state.currentDepthMapPath) {
        addChatMessage('⚠️ Please generate a silhouette first.', 'ai');
        return;
    }

    const modal = document.getElementById('silhouetteEditor');
    const canvas = document.getElementById('editCanvas');
    const ctx = canvas.getContext('2d');
    
    editorState.canvas = canvas;
    editorState.ctx = ctx;
    editorState.history = [];
    
    // 加载当前轮廓图像
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = function() {
        // 设置画布尺寸（保持原始分辨率）
        canvas.width = img.width;
        canvas.height = img.height;
        
        // 绘制图像
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        
        // 保存初始状态
        editorState.history.push(canvas.toDataURL());
        editorState.originalImageSrc = canvas.toDataURL();
        
        updateCanvasInfo('Ready to edit');
    };
    
    img.onerror = function() {
        addChatMessage('⚠️ Failed to load silhouette image.', 'ai');
        closeSilhouetteEditor();
    };
    
    // 构建完整URL
    const imageUrl = modelPreview.src.startsWith('http') 
        ? modelPreview.src 
        : `${API_BASE}${modelPreview.src.replace(API_BASE, '')}`;
    
    img.src = imageUrl + '?t=' + Date.now();
    
    modal.style.display = 'block';
    
    // 绑定画布事件
    setupCanvasEvents();
}

// 关闭编辑器
function closeSilhouetteEditor() {
    const modal = document.getElementById('silhouetteEditor');
    modal.style.display = 'none';
    
    // 清理事件监听
    if (editorState.canvas) {
        editorState.canvas.onmousedown = null;
        editorState.canvas.onmousemove = null;
        editorState.canvas.onmouseup = null;
        editorState.canvas.onmouseleave = null;
        editorState.canvas.ontouchstart = null;
        editorState.canvas.ontouchmove = null;
        editorState.canvas.ontouchend = null;
    }
}

// 设置画布事件监听
function setupCanvasEvents() {
    const canvas = editorState.canvas;
    
    // 鼠标事件
    canvas.onmousedown = startDrawing;
    canvas.onmousemove = draw;
    canvas.onmouseup = stopDrawing;
    canvas.onmouseleave = stopDrawing;
    
    // 触摸事件（移动设备支持）
    canvas.ontouchstart = (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        const mouseEvent = new MouseEvent('mousedown', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    };
    
    canvas.ontouchmove = (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const mouseEvent = new MouseEvent('mousemove', {
            clientX: touch.clientX,
            clientY: touch.clientY
        });
        canvas.dispatchEvent(mouseEvent);
    };
    
    canvas.ontouchend = (e) => {
        e.preventDefault();
        canvas.dispatchEvent(new MouseEvent('mouseup'));
    };
}

function startDrawing(e) {
    editorState.isDrawing = true;
    const rect = editorState.canvas.getBoundingClientRect();
    
    editorState.lastX = (e.clientX - rect.left) * (editorState.canvas.width / rect.width);
    editorState.lastY = (e.clientY - rect.top) * (editorState.canvas.height / rect.height);
    
    editorState.ctx.beginPath();
    editorState.ctx.moveTo(editorState.lastX, editorState.lastY);
    
    updateCanvasInfo('Drawing...');
}

function draw(e) {
    if (!editorState.isDrawing) return;
    
    const canvas = editorState.canvas;
    const ctx = editorState.ctx;
    const rect = canvas.getBoundingClientRect();
    
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);
    
    ctx.lineWidth = editorState.brushSize;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    if (editorState.currentTool === 'pen') {
        ctx.strokeStyle = 'white';
        ctx.globalCompositeOperation = 'source-over';
    } else {
        ctx.strokeStyle = 'black';
        ctx.globalCompositeOperation = 'source-over';
    }
    
    ctx.lineTo(x, y);
    ctx.stroke();
    
    editorState.lastX = x;
    editorState.lastY = y;
}

function stopDrawing() {
    if (editorState.isDrawing) {
        editorState.isDrawing = false;
        editorState.ctx.closePath();
        
        // 保存到历史记录
        editorState.history.push(editorState.canvas.toDataURL());
        if (editorState.history.length > 30) {
            editorState.history.shift();
        }
        
        updateCanvasInfo(`Edits: ${editorState.history.length - 1} steps`);
    }
}
// 选择工具
function selectTool(tool) {
    editorState.currentTool = tool;
    
    document.getElementById('penTool').classList.remove('active');
    document.getElementById('eraserTool').classList.remove('active');
    
    if (tool === 'pen') {
        document.getElementById('penTool').classList.add('active');
        editorState.canvas.style.cursor = 'crosshair';
    } else {
        document.getElementById('eraserTool').classList.add('active');
        editorState.canvas.style.cursor = 'not-allowed';
    }
    
    updateCanvasInfo(`Tool: ${tool === 'pen' ? 'Draw (White)' : 'Erase (Black)'}`);
}

// 更新画笔大小
function updateBrushSize(value) {
    editorState.brushSize = parseInt(value);
    document.getElementById('brushSizeLabel').textContent = value;
}

// 清空画布
function clearCanvas() {
    if (confirm('⚠️ Clear all content? This will make the canvas completely black.')) {
        const ctx = editorState.ctx;
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, editorState.canvas.width, editorState.canvas.height);
        editorState.history.push(editorState.canvas.toDataURL());
        updateCanvasInfo('Canvas cleared');
    }
}

// 撤销
function undoEdit() {
    if (editorState.history.length > 1) {
        editorState.history.pop();
        const previousState = editorState.history[editorState.history.length - 1];
        
        const img = new Image();
        img.onload = function() {
            editorState.ctx.clearRect(0, 0, editorState.canvas.width, editorState.canvas.height);
            editorState.ctx.drawImage(img, 0, 0);
        };
        img.src = previousState;
        
        updateCanvasInfo(`Undo - ${editorState.history.length - 1} steps remain`);
    } else {
        updateCanvasInfo('⚠️ No more undo steps');
    }
}

// 重置到原始图像
function resetToOriginal() {
    if (confirm('🔄 Reset to original silhouette? All edits will be lost.')) {
        if (editorState.originalImageSrc) {
            const img = new Image();
            img.onload = function() {
                editorState.ctx.clearRect(0, 0, editorState.canvas.width, editorState.canvas.height);
                editorState.ctx.drawImage(img, 0, 0);
                editorState.history = [editorState.originalImageSrc];
                updateCanvasInfo('Reset to original');
            };
            img.src = editorState.originalImageSrc;
        }
    }
}
// 更新画布信息
function updateCanvasInfo(text) {
    const infoElement = document.getElementById('canvasInfo');
    if (infoElement) {
        infoElement.textContent = text;
    }
}
// 保存编辑后的轮廓
async function saveEditedSilhouette() {
    showLoading('💾 Saving edited silhouette...');
    
    try {
        // 将画布转换为 Blob
        const blob = await new Promise(resolve => {
            editorState.canvas.toBlob(resolve, 'image/png');
        });
        
        // 先上传编辑后的图像
        const uploadFormData = new FormData();
        uploadFormData.append('file', blob, 'edited_silhouette.png');
        
        const uploadResponse = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: uploadFormData
        });
        
        const uploadData = await uploadResponse.json();
        
        if (!uploadData.success) {
            throw new Error('Failed to upload edited image');
        }
        
        // 获取用户输入的编辑说明
        const instructions = document.getElementById('editInstructions').value.trim() || 'Manual edits applied by user';
        
        // 调用 edit-silhouette API（AI进一步优化）
        const editFormData = new FormData();
        editFormData.append('local_path', uploadData.local_path);
        editFormData.append('instructions', instructions);
        
        const editResponse = await fetch(`${API_BASE}/edit-silhouette`, {
            method: 'POST',
            body: editFormData
        });
        
        const editData = await editResponse.json();
        
        if (editData.success) {
            // 更新状态和预览
            state.currentDepthMapPath = editData.local_path;
            modelPreview.src = `${API_BASE}${editData.image_url}?t=${Date.now()}`;
            modelPreview.style.display = 'block';
            
            addChatMessage('✅ Silhouette updated successfully!', 'ai');
            
            // 清空编辑说明
            document.getElementById('editInstructions').value = '';
            
            closeSilhouetteEditor();
            
            // 自动重新生成3D模型
            addChatMessage('🔄 Regenerating 3D models with updated silhouette...', 'ai');
            await generate3DModels();
            addChatMessage('✅ 3D models regenerated!', 'ai');
        } else {
            throw new Error('Failed to process edited silhouette');
        }
    } catch (error) {
        console.error('Save error:', error);
        addChatMessage('❌ Error saving edited silhouette: ' + error.message, 'ai');
    } finally {
        hideLoading();
    }
}

// 初始化
console.log('App initialized');
