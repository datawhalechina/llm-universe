from flask import Flask

# 创建 Flask 应用实例
app = Flask(__name__)

# 定义路由和视图函数
@app.route('/')
def home():
    return 'Hello, Flask!'

# 启动应用
if __name__ == '__main__':
    app.run(debug=True, port=8081)