import os
import sys
import json
import shutil
from datetime import datetime
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# Import main pipeline từ main.py
from main import initialize_pipeline

app = Flask(__name__, template_folder='../frontend/templates', static_folder='../frontend/static')
CORS(app)

# Configuration cho file upload
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'config/uploaded_tools')
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '16')) * 1024 * 1024  # MB to bytes
ALLOWED_EXTENSIONS = {'json'}

# Tạo upload folder nếu chưa có
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize pipeline instance
pipeline = None

def allowed_file(filename):
    """Kiểm tra file extension có được phép không"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_tools_json(json_data):
    """Validate JSON structure cho tools config"""
    if not isinstance(json_data, list):
        return False, "Config phải là một array"
    
    for i, tool in enumerate(json_data):
        if not isinstance(tool, dict):
            return False, f"Tool {i} phải là object"
        
        # Required fields
        required_fields = ["name", "description", "parameters"]
        for field in required_fields:
            if field not in tool:
                return False, f"Tool {i} thiếu field bắt buộc '{field}'"
        
        # Validate parameters structure
        params = tool.get("parameters", {})
        if not isinstance(params, dict):
            return False, f"Tool {i}: parameters phải là object"
        
        if "properties" not in params:
            return False, f"Tool {i}: parameters phải có 'properties'"
    
    return True, "Valid"

def backup_current_config():
    """Backup config hiện tại trước khi load config mới"""
    current_config_path = "config/tools.json"
    backup_path = f"config/tools_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        if os.path.exists(current_config_path):
            shutil.copy2(current_config_path, backup_path)
            return backup_path
    except Exception as e:
        print(f"Warning: Could not backup config: {e}")
    
    return None

def get_pipeline():
    """Lấy pipeline instance"""
    global pipeline
    if pipeline is None:
        pipeline = initialize_pipeline()
    return pipeline

@app.route('/')
def index():
    """Hiển thị trang chủ"""
    try:
        return render_template('index.html')
    except:
        return f"""
        <div style="text-align: center; padding: 50px; font-family: Arial;">
            <h1>🏛️ NEO RAG API Server</h1>
            <h3>Hệ thống Q&A Pháp luật Việt Nam</h3>
            <p>Frontend templates chưa được cấu hình.</p>
            <hr>
            <h4>📋 API Endpoints:</h4>
            <ul style="list-style: none;">
                <li><b>POST /api/chat</b> - Chat endpoint</li>
                <li><b>GET /api/health</b> - Health check</li>
                <li><b>GET /api/info</b> - System information</li>
            </ul>
            <p><i>API Server đang chạy tại port 8000</i></p>
        </div>
        """

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint cho chat"""
    try:
        # Lấy dữ liệu từ request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        message = data.get('message')
        
        # Validation
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        if not isinstance(message, str):
            return jsonify({'error': 'Message must be a string'}), 400
            
        print(f"Processing chat request - Message: {message[:100]}...")
        
        # Delegate to main pipeline
        current_pipeline = get_pipeline()
        if not current_pipeline:
            return jsonify({'error': 'System not initialized properly'}), 500
            
        response, source = current_pipeline.process_chain(message)
        
        if not response:
            return jsonify({'error': 'No response generated'}), 500
            
        print(f"Chat response generated - Source: {source}, Response: {response[:100]}...")
        
        return jsonify({
            'response': response,
            'source': source,
            'timestamp': datetime.now().isoformat()
        })
        
    except json.JSONDecodeError:
        return jsonify({'error': 'Invalid JSON format'}), 400
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'NEO RAG API',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/info')
def system_info():
    """System information endpoint"""
    return jsonify({
        'service': 'NEO RAG - Vietnamese Legal Q&A System',
        'version': '1.0.0',
        'features': [
            'Function Calling',
            'Document Retrieval',
            'LLM Response Generation',
            'Dynamic Tools Management'
        ],
        'endpoints': {
            'chat': 'POST /api/chat',
            'health': 'GET /api/health',
            'info': 'GET /api/info',
            'tools_upload': 'POST /api/tools/upload',
            'tools_load': 'POST /api/tools/load',
            'tools_list': 'GET /api/tools/list',
            'tools_current': 'GET /api/tools/current'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/tools/upload', methods=['POST'])
def upload_tools_file():
    """Upload tools JSON file"""
    try:
        # Kiểm tra có file trong request không
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Kiểm tra filename
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Kiểm tra extension
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only .json files are allowed'}), 400
        
        # Kiểm tra file size
        if request.content_length > MAX_FILE_SIZE:
            return jsonify({'error': f'File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB'}), 400
        
        # Đọc và validate JSON
        try:
            file_content = file.read()
            json_data = json.loads(file_content.decode('utf-8'))
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Invalid JSON format: {str(e)}'}), 400
        except UnicodeDecodeError:
            return jsonify({'error': 'File encoding error. Please use UTF-8.'}), 400
        
        # Validate tools structure
        is_valid, error_msg = validate_tools_json(json_data)
        if not is_valid:
            return jsonify({'error': f'Invalid tools config: {error_msg}'}), 400
        
        # Tạo tên file với timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_name = secure_filename(file.filename)
        name_without_ext = original_name.rsplit('.', 1)[0]
        new_filename = f"{name_without_ext}_{timestamp}.json"
        
        # Save file
        filepath = os.path.join(UPLOAD_FOLDER, new_filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': new_filename,
            'filepath': filepath,
            'tools_count': len(json_data),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/tools/load', methods=['POST'])
def load_tools_config():
    """Load tools config từ uploaded file vào hệ thống"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'Filename is required'}), 400
        
        # Đường dẫn file
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Kiểm tra file tồn tại
        if not os.path.exists(filepath):
            return jsonify({'error': f'File not found: {filename}'}), 404
        
        # Backup config hiện tại
        backup_path = backup_current_config()
        
        try:
            # Reload pipeline với config mới
            current_pipeline = get_pipeline()
            success = current_pipeline.function_calling.reload_config(filepath)
            
            if success:
                return jsonify({
                    'message': 'Tools config loaded successfully',
                    'config_file': filepath,
                    'backup_file': backup_path,
                    'tools_count': len(current_pipeline.function_calling.tools),
                    'timestamp': datetime.now().isoformat()
                })
            else:
                return jsonify({'error': 'Failed to reload config'}), 500
                
        except Exception as e:
            return jsonify({'error': f'Failed to load config: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Load failed: {str(e)}'}), 500

@app.route('/api/tools/list', methods=['GET'])
def list_uploaded_files():
    """Liệt kê các file tools đã upload"""
    try:
        files = []
        
        # Scan upload folder
        for filename in os.listdir(UPLOAD_FOLDER):
            if filename.endswith('.json'):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                stat = os.stat(filepath)
                
                # Đọc file để lấy thông tin tools
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    tools_count = len(json_data) if isinstance(json_data, list) else 0
                except:
                    tools_count = 0
                
                files.append({
                    'filename': filename,
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / (1024*1024), 2),
                    'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'tools_count': tools_count
                })
        
        # Sort by modified time (newest first)
        files.sort(key=lambda x: x['modified_time'], reverse=True)
        
        return jsonify({
            'files': files,
            'total_files': len(files),
            'upload_folder': UPLOAD_FOLDER,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to list files: {str(e)}'}), 500

@app.route('/api/tools/current', methods=['GET'])
def get_current_tools():
    """Lấy thông tin tools config hiện tại"""
    try:
        current_pipeline = get_pipeline()
        tools = current_pipeline.function_calling.tools
        config_path = current_pipeline.function_calling.tools_config_path
        
        # Thống kê
        local_functions = []
        external_functions = []
        
        for tool in tools:
            if 'endpoint' in tool:
                external_functions.append({
                    'name': tool['name'],
                    'description': tool['description'],
                    'endpoint': tool['endpoint'],
                    'method': tool.get('method', 'POST')
                })
            else:
                local_functions.append({
                    'name': tool['name'],
                    'description': tool['description']
                })
        
        return jsonify({
            'config_path': config_path,
            'total_tools': len(tools),
            'local_functions': local_functions,
            'external_functions': external_functions,
            'local_count': len(local_functions),
            'external_count': len(external_functions),
            'api_base_url': current_pipeline.function_calling.api_base_url,
            'api_configured': bool(current_pipeline.function_calling.api_token),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to get current tools: {str(e)}'}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(app.static_folder, filename)

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/api/chat', '/api/health', '/api/info'],
        'timestamp': datetime.now().isoformat()
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed',
        'timestamp': datetime.now().isoformat()
    }), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500

if __name__ == "__main__":
    print("   🏛️  NEO RAG API Server Starting...")
    print("   📋 Main APIs:")
    print("   • GET  /              - Web interface")
    print("   • POST /api/chat      - Chat API")
    print("   • GET  /api/health    - Health check")
    print("   • GET  /api/info      - System info")
    print("")
    print("   🔧 Tools Management APIs:")
    print("   • POST /api/tools/upload  - Upload JSON tools")
    print("   • POST /api/tools/load    - Load tools config")
    print("   • GET  /api/tools/list    - List uploaded files")
    print("   • GET  /api/tools/current - Current tools config")
    print("")
    print("   🌐 Server running at: http://localhost:8000")
    print("   📁 Upload folder:", UPLOAD_FOLDER)
    print("   📏 Max file size:", MAX_FILE_SIZE // (1024*1024), "MB")
    
    app.run(host="0.0.0.0", port=8000, debug=True)