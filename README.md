<h1 align="center">Beyond The FRONTIER</h1>

<p align="center">
    <a href="https://www.tensorflow.org/">
        <img src="https://img.shields.io/badge/Tensorflow-1.13-green" alt="Vue2.0">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Author-JialiCui-blueviolet" alt="Author">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Email-cuijiali961224@gmail.com-blueviolet" alt="Author">
    </a>
    <a href="https://www.stevens.edu/">
        <img src="https://img.shields.io/badge/College-SIT-green" alt="Vue2.0">
    </a>
</p>








# Attack Paper



## **Defense-VAE: A Fast and Accurate Defense**

[Paper Here](https://arxiv.org/pdf/1812.06570.pdf)

****

* <div align="center">
      <img src="https://github.com/CuiJiali-CV/attack-paper-note/raw/main/defense-vae/train.png" height="400" width="1000" >
  </div>

* <div align="center">
      <img src="https://github.com/CuiJiali-CV/attack-paper-note/raw/main/defense-vae/test.png" height="200" width="1000" >
  </div>

* <div align="center">
      <img src="https://latex.codecogs.com/gif.latex?\pounds&space;_{Defense-VAE}=-E_{q(z|\hat{x})}[logp(x|z)]&plus;D_{KL}(q(z|\hat{x})||p(z))" >
  </div>

* **It can also train the whole pipeline end to end from scratch or finetuning from pre-trained VAE and CNN classififier by optimizing the joint loss function**

* <div align="center">
      <img src="https://latex.codecogs.com/gif.latex?\pounds&space;_{End-to-End}=&space;\pounds&space;_{Defense-VAE}&space;&plus;&space;\lambda\pounds&space;_{Cross-Entropy}" >
  </div>


  ### Questions

  - This VAE is trained with adversarial examples. Why not just train with nature images ? 
    - **During the reconstructing, Encoder is not able to denoise the images by inferring a purified latent z ?**
    - Given Encoder of VAE is a top-down model, which is kind of sensitive for any little change in the input, **can we use MCMC to replace Encoder in order to denoise the image. (if so, we could naturely train the model and denoise the image by inferring a purified latent z )**
  - This is telling us that adversarially trained VAE ( but do MSE with nature images) could denoise the image. 
    - **Then, intuitively, we could use any reconstruction based model to do such thing ?**

  

  

## **Purifying Adversarial Perturbation with Adversarially Trained Auto-encoders**

  [Paper Here](https://arxiv.org/pdf/1905.10729.pdf)

****

- <div align="center">
      <img src="https://latex.codecogs.com/gif.latex?min_{\o&space;}E_{(x,y)\in&space;\chi&space;}[max_{\delta\in&space;S}L(\theta;AE_{\o}(x&plus;\delta),y)&plus;\lambda&space;L_{Cross-Entropy}(AE_{\o}(x),x)]" >
  </div>

- **θ is fixed ( pre-trained classifier), adversarially train VAE only.**





## **PuVAE: A Variational Autoencoder to Purify Adversarial Examples**

   [Paper Here](https://arxiv.org/pdf/1903.00585.pdf)

****

- <div align="center">
      <img src="https://github.com/CuiJiali-CV/attack-paper-note/raw/main/PuVAE/Train.png" height="200" width="600" >
  </div>

- ```python
  Train Algorithm:
      For n epochs:
        mean,std = Encoder(x,y)
          Loss += KL(mean, std)
          
          z = reparameter(mean, std)
          x_hat = Decoder(z, y)
          Loss += MSE(x_hat, x)
          
          y_s = Source_Classifier(x_hat)
          Loss += Cross-Entropy(y_s, y)
          
          Update weights
      
  ```

- <div align="center">
      <img src="https://github.com/CuiJiali-CV/attack-paper-note/raw/main/PuVAE/test.png" height="300" width="800" >
  </div>

- ```python
  Inference Algorithm:
  	x_hat, ys = PuVAE(x_adv,y)
  	y* = argmin(RMSE([x_hat,ys],[x_adv,y]))
  	
  	Then x_hat_y* is the purified image
  	Do Target_Classifier(x_hat_y*)
  ```

  ### Questions

  - This VAE is trained with a classifier. The difference is this model use nature images to train VAE, although it has labels as input, which is similar to cVAE.  Therefore, a intuitive question is that can we train it without a classifier ?
    - Some paper (Purifying Adversarial Perturbation with Adversarially Trained Auto-encoders) mention this and say the reason it always includes a classifier is because it will decrease the standard accuracy when it is trained without a classifier.
    - I think it may be motivation of this paper. Since unsupervised VAE is trained with a classifier, and they are mostly under adversarial training. Why not try a supervised VAE and train it naturely ?
    - **Does label empower this VAE to train naturely ?** 
    - **Does it means, that the Encoder of an unlabeled VAE could not infer a purifed latent is because it lack of some extra information to boost its capability ?**

  

  



  


## Author

```javascript
var iD = {
  name  : "Jiali Cui",
  
  bachelor: "Harbin Institue of Technology",
  master : "Stevens Institute of Technology",
  
  Interested: "CV, ML"
}
```
