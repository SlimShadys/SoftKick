# SoftKick

## Requirements and installation
Due to specific requirements in the RLGym library, it is necessary to manually install particular versions of the required libraries.

Here's the steps:
1. Install Python version 3.9.x
2. Install Visual C++ 14.0 or greater from https://visualstudio.microsoft.com/visual-cpp-build-tools/
3. Run `pip install setuptools==65.5.0 pip==21` as gym 0.21 installation is broken with more recent versions.
4. Run `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
5. Run `pip install wheel==0.38.4`
6. Run `pip install swig`
7. Run `pip install gym[box2d]`
8. Run `pip install rlgym rlgym-tools`
9.  Run `pip install rocketsim git+https://github.com/AechPro/rocket-league-gym-sim@main git+https://github.com/AechPro/rlgym-ppo`