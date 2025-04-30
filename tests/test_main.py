from typer.testing import CliRunner
from zoom_earth_cli.main import app
import pytest

runner = CliRunner()

@pytest.mark.parametrize("slider_path,bg_path,expected", [
    ("tests/ROTATE/228_blue.png", "tests/ROTATE/bg_blue.png", "228"),
    ("tests/ROTATE/121_blue.png", "tests/ROTATE/bg_blue.png", "121"),
    ("tests/ROTATE/102_green.png", "tests/ROTATE/bg_green.png", "102"),
    # 可以继续添加更多测试用例
])
def test_rotate(slider_path, bg_path, expected, capsys):  # 添加 capsys 参数
    result = runner.invoke(
        app,
        [
            "rotate",
            "--slider-path", slider_path,
            "--bg-path", bg_path
        ]
    )
    
    # 获取被捕获的输出
    captured = capsys.readouterr()
    print_output = captured.out
    
    # 断言返回值
    assert result.exit_code == 0
    # 断言预期结果出现在标准输出
    assert expected in result.stdout

@pytest.mark.parametrize("bg_path,expected", [
    ("tests/CONCAT/312_blue.png", "312"),
    ("tests/CONCAT/470_green.png", "470"),
    ("tests/CONCAT/225_blue.png", "225"),
    ("tests/CONCAT/409_green.png", "409"),
    ("tests/CONCAT/374_blue.png", "374"),
    # 可以继续添加其他测试用例...
])
def test_concat(bg_path, expected, capsys):
    result = runner.invoke(
        app,
        [
            "concat",
            "--bg-path", bg_path
        ]
    )
    
    # 捕获普通 print 输出
    captured = capsys.readouterr()
    print_output = captured.out

    # 验证基础断言
    assert result.exit_code == 0, "命令应该成功执行"
    # 验证预期日志
    assert expected in result.stdout

@pytest.mark.parametrize("slider_path,bg_path,expected", [
    ("tests/SLIDER/template_68.png", "tests/SLIDER/bg_68.jpeg", "149"), # 68 ?
    ("tests/SLIDER/template_127.png", "tests/SLIDER/bg_127.jpeg", "127"), # 64
    ("tests/SLIDER/template_309.png", "tests/SLIDER/bg_309.jpeg", "309"), # 149
    # 可以继续添加更多测试用例
])
def test_slider(slider_path, bg_path, expected, capsys):  # 添加 capsys 参数
    result = runner.invoke(
        app,
        [
            "slider",
            "--slider-path", slider_path,
            "--bg-path", bg_path
        ]
    )
    
    # 获取被捕获的输出
    captured = capsys.readouterr()
    print_output = captured.out
    
    # 断言返回值
    assert result.exit_code == 0
    # 断言预期结果出现在标准输出
    assert expected in result.stdout