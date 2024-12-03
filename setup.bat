@echo off
echo Creating virtual environment...
python -m venv tf_test_env

echo Activating virtual environment...
call tf_test_env\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

echo Setup complete! To activate the environment, run:
echo tf_test_env\Scripts\activate
pause 