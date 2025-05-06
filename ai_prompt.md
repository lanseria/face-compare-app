根据上面的代码，使用 fastapi，我已经安装 fastapi
完成 src/face_compare_app/cli.py 中 命令 server 这个方法
首先完成这几个api的壳的实现，先不实现具体逻辑，但可以接收参数，创建单独的文件夹与 py 文件，来处理 server 路由请求
1. 基础比对 API
POST /api/v1/compare
Content-Type: multipart/form-data

参数：
- image1 : 文件 (必选)
- image2 : 文件 (必选)
- threshold : 数值 (可选，相似度阈值)

响应：
{
  "similarity": 0.92,
  "is_match": true,
  "elapsed_ms": 120
}
2. 实时摄像头比对（WebSocket）
GET /api/v1/live/ws?reference_id=user123
Headers:
- Upgrade: websocket

数据流：
客户端发送实时视频帧 -> 服务端返回比对结果
3. 人脸信息插入特征库
POST /api/v1/faces
Content-Type: multipart/form-data

参数：
- image : 文件 (必选)
- id : 字符串 (必选)
- name : 字符串 (可选)
- meta : JSON字符串 (可选)

响应：
{
  "face_id": "user123",
  "feature_size": 512,
}
4.  特征库搜索
POST /api/v1/search
Content-Type: multipart/form-data

参数：
- image : 文件 (必选)
- top_k : 整数 (可选，默认3)
- threshold : 数值 (可选)

响应：
{
  "results": [
    {
      "face_id": "user123",
      "name": "John Doe",
      "similarity": 0.95,
      "meta": {"age":30}
    }
  ],
  "search_time_ms": 45
}
5. 实时摄像头查询
GET /api/v1/live-search
Headers:
- Upgrade: websocket

数据流：
客户端发送实时视频帧 -> 服务端返回结果
// On Successful Match:
{
  "status": "match_found",
  "match": {
    "face_id": "user123", // Matched Face ID from DB
    "name": "John Doe",    // Optional: Name from DB
    "similarity": 0.95,    // Similarity score
    "meta": {"age": 30}     // Optional: Metadata from DB (if return_meta is true)
  },
  // Optional fields for client-side rendering/info:
  "detection_box": [150, 100, 250, 220], // [x1, y1, x2, y2] coordinates in the frame
  "processed_frame_timestamp_ms": 1678886400123 // Server timestamp when processed
}
// On No Face Detected (Optional): You might choose not to send anything to save bandwidth, or send a status periodically:
{
  "status": "no_face_detected"
}
// On Face Detected, No Match (Optional): Similar to above, sending nothing is often preferred.
{
  "status": "no_match_found",
  // Optional: Include detection box even if no match
  "detection_box": [150, 100, 250, 220]
}
// On Error (e.g., invalid frame format):
{
  "status": "error",
  "message": "Invalid frame data received."
}