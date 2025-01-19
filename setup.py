from setuptools import setup, find_packages


setup(name='Multi Queue',
      version='1.0.0',
      description='Multi Queue for Unsupervised Person Re-Identification',
      author='Shengyong Xie',
      author_email='2416057194@qq.com',
      # url='',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Object Re-identification'
      ])
