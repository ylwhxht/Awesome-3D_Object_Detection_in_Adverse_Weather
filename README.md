# Paper List about 3D Object Detection in Adverse Weather

<p align="center">
<img src="./images/detection.gif", width="1000">
</p>

*图摘自[Kradar](https://github.com/kaist-avelab/K-Radar)*

## 🏠 About

Hi, this is the paper list about 3D Object Detection in Adverse Weather. It is mainly derived from papers that are considered meaningful during the last 3 years of research. Hopefully, it can provide some help for newbie researchers.
﻿
It may be brief and there are some omissions, welcome to suggest them in the Issue, we will update them timely~. 

If you think it's useful, come up with one ⭐， Thank you ^_^

## :collision: Update Log 
* [2025.3.8] We release the first version of the paper list for 3D Object Detection in Adverse Weather.

## <a id="table-of-contents">📚 Table of Contents </a>
* [Survey](#surveys)
* [Dataset](#dataset)
* [Weather Quantitative Analysis](#eva)
* [LiDAR Adverse Weather Simulation](#sim)
* [LiDAR Denoiser](#lidarde)
* [LiDAR-based/with Camera Detector](#lidar)
* [4D Radar-based/with Camera Detector](#radar)
* [LiDAR+3D Radar Fusion Detector](#l3r)
* [LiDAR+4D Radar Fusion Detector](#l4r)
* [with Cooperative Perception](#icp)
## <a id="surveys"> Surveys <a href="#table-of-contents">🔝</a> </a>


### *2022*
* **Perception and Sensing for Autonomous Vehicles Under Adverse Weather Conditions: A Survey**, ISPRS 2022                  
[[paper](https://arxiv.org/abs/2112.08936)]


* **3D Object Detection for Autonomous Driving: A Survey**, Pattern Recognition 2022                  
[[paper](https://arxiv.org/abs/2106.10823)]  



### *2023*

* **Performance and Challenges of 3D Object Detection Methods in Complex Scenes for Autonomous Driving**, TIV 2023                  
[[paper](https://arxiv.org/abs/2206.09474#:~:text=This%20paper%20reviews%20the%20advances%20in%203D%20object,detection%20and%20discuss%20the%20challenges%20in%20this%20task.)]     

* **Survey on LiDAR Perception in Adverse Weather Conditions**, IV 2023                  
[[paper](https://arxiv.org/abs/2304.06312)]     




### *2024*

* **Object Detection in Autonomous Vehicles under Adverse Weather: A Review of Traditional and Deep Learning Approaches**, Algorithms 2024                  
[[paper](https://ieeexplore.ieee.org/abstract/document/10839256)]  

* **Perception Methods for Adverse Weather Based on Vehicle Infrastructure Cooperation System: A Review**, Sensors 2024                  
[[paper](https://www.mdpi.com/1999-4893/17/3/103)]     

* **Robustness-Aware 3D Object Detection in Autonomous Driving: A Review and Outlook**, TITS 2024                  
[[paper](https://ieeexplore.ieee.org/abstract/document/10637966)]   
 

### *2025*


* **LiDAR Denoising Methods in Adverse Environments: A Review**, Sensors 2025                  
[[paper](https://ieeexplore.ieee.org/abstract/document/10839256)]     


----
## <a id="dataset"> Datasets <a href="#table-of-contents">🔝</a> 

### *2021*

 * **[DENSE(STF)]: Seeing Through Fog Without Seeing Fog: Deep Multimodal Sensor Fusion in Unseen Adverse Weather**, CVPR 2020                  
[[paper](https://arxiv.org/abs/1902.08913)] [[data](https://www.uni-ulm.de/en/in/driveu/projects/dense-datasets)]   

 * **[WOD-DA]: Waymo Open Dataset Domain Adaptation**, 2020                  
[[data](https://waymo.com/open/challenges/2020/domain-adaptation/)]  

### *2022*

* **[CADC]: Canadian Adverse Driving Conditions Dataset**, IJRR 2021                  
[[paper](https://www.mdpi.com/1424-8220/24/2/374)] [[data](https://gitee.com/yangmissionyang/cadc_devkit)]    

### *2023*

* **[Kradar]: K-radar: 4d radar object detection for autonomous driving in various weather conditions**, NIPS 2022                  
[[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/185fdf627eaae2abab36205dcd19b817-Paper-Datasets_and_Benchmarks.pdf)] [[code&data](https://github.com/kaist-avelab/K-Radar)]  

* **[WADS]: Winter adverse driving dataset for autonomy in inclement winter weather**, Optical Engineering 2023                  
[[paper](https://www.spiedigitallibrary.org/journals/optical-engineering/volume-62/issue-3/031207/Winter-adverse-driving-dataset-for-autonomy-in-inclement-winter-weather/10.1117/1.OE.62.3.031207.full)] [[code&data](https://bitbucket.org/autonomymtu/wads/src/master/)]  

* **[SemanticSpray++]: SemanticSpray++: A Multimodal Dataset for Autonomous Driving in Wet Surface Conditions**, IV 2024                  
[[paper](https://ieeexplore.ieee.org/abstract/document/10588458)] [[code&data](https://semantic-spray-dataset.github.io/)]  

### *2024*

* **Is Your LiDAR Placement Optimized for 3D Scene Understanding?**, NIPS 2024                  
[[paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/3dbb8b6b5576b85afb3037e9630812dc-Paper-Conference.pdf)] [[code&data](https://github.com/ywyeli/Place3D)]  


----
## <a id="eva"> Weather Quantitative Analysis<a href="#table-of-contents">🔝</a> 

### *2009*

* **Performance of Laser and Radar Ranging Devices in Adverse Environmental Conditions**, Journal of Field Robotics 2009                  
[[paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e8fe17bfa1615afe1f64007a7d7fc797ae7a4624)]
### *2018*

* **A Benchmark for Lidar Sensors in Fog: Is Detection Breaking Down?**, IV 2018                  
[[paper](https://arxiv.org/pdf/1912.03251)]

### *2020*

* **Analysis of automotive lidar sensor model considering scattering effects in regional rain environments**, Access 2020                  
[[paper](https://ieeexplore.ieee.org/iel7/6287639/6514899/09097838.pdf)]  
### *2021*

* **A Quantitative Analysis of Point Clouds from Automotive Lidars Exposed to Artificial Rain and Fog**, Atmosphere 2021                  
[[paper](https://www.mdpi.com/2073-4433/12/6/738)]
### *2022*

* **Measuring the Influence of Environmental Conditions on Automotive Lidar Sensors**, Sensors 2022                  
[[paper](https://www.mdpi.com/1424-8220/22/14/5266)]  

* **Camera and LiDAR analysis for 3D object detection in foggy weather conditions**, ICPRS 2022                  
[[paper](https://ieeexplore.ieee.org/abstract/document/9854073)]  
### *2023*


* **Benchmarking Robustness of 3D Object Detection to Common Corruptions in Autonomous Driving**,CVPR 2023                  
[[paper](https://arxiv.org/abs/2303.11040)] [[code](https://github.com/thu-ml/3D_Corruptions_AD)]  
### *2024*


* **Effect of Fog Particle Size Distribution on 3D Object Detection Under Adverse Weather Conditions**, Arxiv 2024                  
[[paper](https://arxiv.org/abs/2408.01085)]





----
## <a id="sim"> LiDAR Adverse Weather Simulation<a href="#table-of-contents">🔝</a> 
### *2018*

* **[FogSimulation]: A Benchmark for Lidar Sensors in Fog: Is Detection Breaking Down?**, IV 2018                  
[[paper](https://arxiv.org/pdf/1912.03251)]
### *2020*

 * **[Fog Simulation]: Seeing Through Fog Without Seeing Fog: Deep Multimodal Sensor Fusion in Unseen Adverse Weather**, CVPR 2020                  
[[paper](https://arxiv.org/abs/1902.08913)] [[code](https://www.uni-ulm.de/en/in/driveu/projects/dense-datasets)]   
### *2021*

 * **[Fog Simulation]: Fog Simulation on Real LiDAR Point Clouds for 3D Object Detection in Adverse Weather**, ICCV 2021                  
[[paper](https://openaccess.thecvf.com/content/ICCV2021/html/Hahner_Fog_Simulation_on_Real_LiDAR_Point_Clouds_for_3D_Object_ICCV_2021_paper.html)] [[code](https://github.com/MartinHahner/LiDAR_fog_sim)]   

* **[Rain Simulation]: Lidar Light Scattering Augmentation (LISA): Physics-based Simulation of Adverse Weather Conditions for 3D Object Detection**, Arxiv  2021                  
[[paper](https://arxiv.org/abs/2107.07004)] [[code](https://github.com/velatkilic/LISA)]    
### *2022*

 * **[Snow Simulation]: https://arxiv.org/abs/2203.15118**, CVPR 2022                  
[[paper](https://arxiv.org/abs/2203.15118)] [[code](https://github.com/SysCV/LiDAR_snow_sim)]    

* **[Spray Simulation]: Reconstruction and Synthesis of Lidar Point Clouds of Spray**, RAL 2022                  
[[paper](https://ieeexplore.ieee.org/abstract/document/9705289)] [[code](https://github.com/Wachemanston/Reconstruction-and-Synthesis-of-Lidar-Point-Clouds-of-Spray)]  
### *2023*

* **[Various Simulation]: Benchmarking Robustness of 3D Object Detection to Common Corruptions in Autonomous Driving**, CVPR 2023                  
[[paper](https://arxiv.org/abs/2303.11040)] [[code](https://github.com/thu-ml/3D_Corruptions_AD)]  

* **[Snow Simulation]: LiDAR Point Cloud Translation Between Snow and Clear Conditions Using Depth Images and GANs**, IV 2023                  
[[paper](https://ieeexplore.ieee.org/abstract/document/10186814)]



* **[Various Simulation]: Robo3D: Towards Robust and Reliable 3D Perception against Corruptions**, ICCV 2023                  
[[paper](http://openaccess.thecvf.com/content/ICCV2023/papers/Kong_Robo3D_Towards_Robust_and_Reliable_3D_Perception_against_Corruptions_ICCV_2023_paper.pdf)] [[code](https://github.com/ldkong1205/Robo3D.)]  

* **[Snow Simulation]: L-DIG: A GAN-Based Method for LiDAR Point Cloud Processing under Snow Driving Conditions**, Sensors 2023                  
[[paper](https://www.mdpi.com/2072-4292/16/12/2247)]
### *2024*

* **[Snow Simulation]: LiDAR Point Cloud Augmentation for Adverse Conditions Using Conditional Generative Model**, Remote Sens. 2024                  
[[paper](https://www.mdpi.com/2072-4292/16/12/2247)]

* **[Rain Simulation]: Sunshine to Rainstorm: Cross-Weather Knowledge Distillation for Robust 3D Object Detection**, AAAI 2024                  
[[paper](https://arxiv.org/abs/2402.18493)] [[code](https://github.com/ylwhxht/SRKD-DRET)]  
### *2025*

* **[Snow Simulation]: Adverse Weather Conditions Augmentation of LiDAR Scenes with Latent Diffusion Models**, Arxiv. 2025                  
[[paper](https://arxiv.org/pdf/2501.01761)]


----
## <a id="lidarde"> LiDAR Denoiser<a href="#table-of-contents">🔝</a> 
### *2018*

 * **De-noising of lidar point clouds corrupted by snowfall**, CRV 2018                  
 [[paper](https://ieeexplore.ieee.org/abstract/document/8575761)]

### *2020*

 * **Fast and Accurate Desnowing Algorithm for LiDAR Point Clouds**, Access 2020                  
 [[paper](https://ieeexplore.ieee.org/abstract/document/9180326)]   

 * **CNN-based Lidar Point Cloud De-Noising in Adverse Weather**, RAL 2020                  
[[paper](https://arxiv.org/pdf/1912.03874)] [[code](https://github.com/rheinzler/PointCloudDeNoising)]   


### *2021*


 * **DSOR: A Scalable Statistical Filter for Removing Falling Snow from LiDAR Point Clouds in Severe Winter Weather**, Arxiv 2021                  
 [[paper](https://arxiv.org/abs/2109.07078)] [[code](https://bitbucket.org/autonomymtu/dsor_filter/src/master/)]





### *2022*

 * **LiSnowNet: Real-time Snow Removal for LiDAR Point Cloud**, IROS 2022                  
 [[paper](https://arxiv.org/abs/2211.10023)]   

 * **De-snowing LiDAR Point Clouds With Intensity and Spatial-Temporal Features**, ICRA 2022                  
 [[paper](https://ieeexplore.ieee.org/document/9812241)]   

 * **A Scalable and Accurate De-Snowing Algorithm for LiDAR Point Clouds in Winter**, Remote Sens. 2022                  
 [[paper](https://www.mdpi.com/2072-4292/14/6/1468)]   
  * **AdverseNet: A LiDAR Point Cloud Denoising Network for Autonomous Driving in Rainy, Snowy, and Foggy Weather**, ICUS. 2022                  
 [[paper](https://ieeexplore.ieee.org/abstract/document/9986567)] [[code](https://github.com/Naclzno/AdverseNet)]

 * **LiSnowNet: Real-time Snow Removal for LiDAR Point Clouds**, IROS 2022                  
 [[paper](https://ieeexplore.ieee.org/abstract/document/9982248)] [[code](https://github.com/umautobots/lisnownet?search=1)]


 * **4denoisenet: Adverse weather denoising from adjacent point clouds**, RAL. 2022                  
 [[paper](https://www.mdpi.com/2072-4292/14/6/1468)]  [[code](https://github.com/alvariseppanen/4DenoiseNet)]


 * **Adaptive Two-Stage Filter for De-snowing LiDAR Point Clouds**, ICCRI 2022                  
 [[paper](https://ieeexplore.ieee.org/abstract/document/10223164)]



### *2023*

 * **RGOR: De-noising of LiDAR point clouds with reflectance restoration in adverse weather**, ICTC. 2023                  
 [[paper](https://ieeexplore.ieee.org/abstract/document/10392388)]



 * **DCOR: Dynamic Channel-Wise Outlier Removal to De-Noise LiDAR Data Corrupted by Snow**, ICIP 2023                  
 [[paper](https://ieeexplore.ieee.org/abstract/document/10401021)]

 * **GAN Inversion Based Point Clouds Denoising in Foggy Scenarios for Autonomous Driving* **, ICDL 2023                  
 [[paper](https://ieeexplore.ieee.org/abstract/document/10364496)]


### *2024*

 * **Denoising Point Clouds with Intensity and Spatial Features in Rainy Weather**, TITS 2024                  
 [[paper](https://ieeexplore.ieee.org/abstract/document/10223164)]


 * **RGB-LiDAR sensor fusion for dust de-filtering in autonomous excavation applications**, Automation in Construction 2024                  
 [[paper](https://www.sciencedirect.com/science/article/pii/S0926580524005867#:~:text=In%20this%20article%2C%20a%20light%20detection%20and%20ranging,model%20was%20developed%20to%20filter%20airborne%20dust%20particles.)]
 
* **TripleMixer: A 3D Point Cloud Denoising Model for Adverse Weather**, Arxiv 2024                  
 [[paper](https://arxiv.org/abs/2408.13802)]  [[code](https://github.com/Grandzxw/TripleMixer/)]


 * **An improved point cloud denoising method in adverse weather conditions based on PP-LiteSeg network**, PeerJ Computer Science 2024                  
 [[paper](https://peerj.com/articles/cs-1832.pdf)]

 * **Denoising Framework Based on Multiframe Continuous Point Clouds for Autonomous Driving LiDAR in Snowy Weather**, Sensors 2024                  
 [[paper](https://ieeexplore.ieee.org/abstract/document/10418150)] [[code](https://github.com/Naclzno/TOR)]

 * **Dust De-Filtering in LiDAR Applications With Conventional and CNN Filtering Methods**, Sensors 2024                  
 [[paper](https://ieeexplore.ieee.org/abstract/document/10423020)]


 * **AdWeatherNet: Adverse Weather Denoising with Point Cloud Spatiotemporal Attention**, VCIP 2024                  
 [[paper](https://ieeexplore.ieee.org/abstract/document/10849938)] [[code](https://github.com/Naclzno/AdverseNet)]

 * **3D-UnOutDet: A Fast and Efficient Unsupervised Snow Removal Algorithm for 3D LiDAR Point Clouds**, Authorea Preprints 2024                  
 [[paper](https://www.techrxiv.org/doi/full/10.36227/techrxiv.172865969.94242670)] [[code](https://github.com/sporsho/3DUnOutDet)]
### *2025*


 * **Semantic Segmentation Based Rain and Fog Filtering Only by LiDAR Point Clouds**, Sensors. 2025                  
 [[paper](https://ieeexplore.ieee.org/abstract/document/10832503)]




----




## <a id="lidar"> LiDAR-based/with Camera Detector<a href="#table-of-contents">🔝</a> 

### *2020*

* **1st Place Solution for Waymo Open Dataset Challenge - 3D Detection and Domain Adaptation**, Arxiv 2020                  
[[paper](https://arxiv.org/pdf/2006.15505)]  


### *2021*

* **SPG: Unsupervised Domain Adaptation for 3D Object Detection via Semantic Point Generation**, CVPR 2021                  
[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_SPG_Unsupervised_Domain_Adaptation_for_3D_Object_Detection_via_Semantic_ICCV_2021_paper.pdf)]  
### *2022*

* **Rethinking LiDAR Object Detection in adverse weather conditions**, ICRA 2022                  
[[paper](https://ieeexplore.ieee.org/abstract/document/9812039)]  

 * **Towards Robust 3D Object Detection In Rainy Conditions** ITSC 2022                  
[[paper](https://www.mdpi.com/2072-4292/16/12/2247/pdf)] [[code](https://github.com/julycrow/3D_Object_Detection_Winter)]

* **LossDistillNet: 3D Object Detection in Point Cloud Under Harsh Weather Conditions**,  Access 2022                  
[[paper](https://arxiv.org/abs/2303.11040)]

* **Robust 3D Object Detection in Cold Weather Conditions**,IV 2022                  
[[paper](https://ieeexplore.ieee.org/abstract/document/9827398)]  

* **Robust-FusionNet: Deep Multimodal Sensor Fusion for 3-D Object Detection Under Severe Weather Conditions**,  TIM 2022                  
[[paper](https://ieeexplore.ieee.org/abstract/document/9831988)]  

### *2023*

* **A Point Cloud-based 3D Object Detection Method for Winter Weather**, ISCER 2023                  
[[paper](https://ieeexplore.ieee.org/abstract/document/10266225)]



* **Source-free Unsupervised Domain Adaptation for 3D Object Detection in Adverse Weather**, ICRA 2023                  
[[paper](https://ieeexplore.ieee.org/abstract/document/10161341)] [[code](https://github.com/deeptibhegde/UncertaintyAwareMeanTeacher)]  


* **Enhancing Lidar-based Object Detection in Adverse Weather using Offset Sequences in Time**, ICECET 2023                  
[[paper](https://arxiv.org/pdf/2401.09049)]

### *2024*

 * **Geometric information constraint 3D object detection from LiDAR point cloud for autonomous vehicles under adverse weather**, Transportation research part C: emerging technologies 2024                  
[[paper](https://www.sciencedirect.com/science/article/pii/S0968090X24000767)]   

* **Sunshine to Rainstorm: Cross-Weather Knowledge Distillation for Robust 3D Object Detection**, AAAI 2024                  
[[paper](https://arxiv.org/abs/2402.18493)] [[code](https://github.com/ylwhxht/SRKD-DRET)]  

* **SAMFusion: Sensor-Adaptive Multimodal Fusion for 3D Object Detection in Adverse Weather**, ECCV 2024                  
[[paper](https://link.springer.com/chapter/10.1007/978-3-031-73030-6_27)]  [[code](https://light.princeton.edu/publication/samfusion/)]



* **LiDAR Point Cloud Augmentation for Adverse Conditions Using Conditional Generative Model**, Remote Sensing 2024                  
[[paper](https://www.mdpi.com/2072-4292/16/12/2247/pdf)]





### *2025*

* **AWARDistill: Adaptive and robust 3D object detection in adverse conditions through knowledge distillation**,Expert Systems with Applications, 2025                  
[[paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417424028999)] 


* **3D vision object detection for autonomous driving in fog using LiDaR**, Simulation Modelling Practice and Theory 2025                  
[[paper](https://www.sciencedirect.com/science/article/abs/pii/S1569190X25000243)]





----
## <a id="radar"> 4D Radar-based/with Camera Detector <a href="#table-of-contents">🔝</a> 
### *2022*

* **[RTNH]: K-radar: 4d radar object detection for autonomous driving in various weather conditions**, NIPS 2022                  
[[paper](https://proceedings.neurips.cc/paper_files/paper/2022/file/185fdf627eaae2abab36205dcd19b817-Paper-Datasets_and_Benchmarks.pdf)] [[code&data](https://github.com/kaist-avelab/K-Radar)]  
### *2024*

* **TL-4DRCF: A Two-Level 4-D Radar–Camera Fusion Method for Object Detection in Adverse Weather**, Sensors 2024                  
[[paper](https://ieeexplore.ieee.org/document/10491101)] 



----
## <a id="l3r"> LiDAR+3D Radar Fusion Detector<a href="#table-of-contents">🔝</a> 
### *2020*

 * **Seeing Through Fog Without Seeing Fog: Deep Multimodal Sensor Fusion in Unseen Adverse Weather**, CVPR 2020                  
[[paper](https://arxiv.org/abs/1902.08913)] [[code](https://www.uni-ulm.de/en/in/driveu/projects/dense-datasets)]  
### *2021*

* **Robust Multimodal Vehicle Detection in Foggy Weather Using Complementary Lidar and Radar Signals**, CVPR 2021                  
[[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Qian_Robust_Multimodal_Vehicle_Detection_in_Foggy_Weather_Using_Complementary_Lidar_CVPR_2021_paper.pdf)] [[code](https://github.com/qiank10/MVDNet)]  
### *2022*

* **Modality-Agnostic Learning for Radar-Lidar Fusion in Vehicle Detection**, CVPR 2022
[[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Modality-Agnostic_Learning_for_Radar-Lidar_Fusion_in_Vehicle_Detection_CVPR_2022_paper.pdf)] 
### *2023*

* **ST-MVDNET++: IMPROVE VEHICLE DETECTION WITH LIDAR-RADAR GEOMETRICAL AUGMENTATION VIA SELF-TRAINING**, ICASSP 2023                  
[[paper](https://ieeexplore.ieee.org/abstract/document/10096041)]  [[code](https://github.com/qiank10/MVDNet)]   

* **Bi-LRFusion: Bi-Directional LiDAR-Radar Fusion for 3D Dynamic Object Detection**, CVPR 2023                  
[[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Bi-LRFusion_Bi-Directional_LiDAR-Radar_Fusion_for_3D_Dynamic_Object_Detection_CVPR_2023_paper.pdf)] 
### *2024*

 * **3D Object Detection Algorithm in Adverse Weather Conditions Based on LiDAR-Radar Fusion**, CCC 2024                  
[[paper](https://ieeexplore.ieee.org/abstract/document/10661603)] 



* **RaLiBEV: Radar and LiDAR BEV Fusion Learning for Anchor Box Free Object Detection Systems**, TCSVT 2024                  
[[paper](https://arxiv.org/abs/2211.06108#:~:text=In%20this%20paper%2C%20we%20propose%20a%20bird%27s-eye%20view,radar%20range-azimuth%20heatmap%20and%20the%20LiDAR%20point%20c)]  [[code](https://github.com/yyxr75/RaLiBEV)]  

* **SAMFusion: Sensor-Adaptive Multimodal Fusion for 3D Object Detection in Adverse Weather**, ECCV 2024                  
[[paper](https://link.springer.com/chapter/10.1007/978-3-031-73030-6_27)]  [[code](https://light.princeton.edu/publication/samfusion/)]

* **TransFusion: Multi-Modal Robust Fusion for 3D Object Detection in Foggy Weather Based on Spatial Vision Transformer**, TITS 2024                  
[[paper](https://ieeexplore.ieee.org/abstract/document/10591357)] 


----
## <a id="l4r"> LiDAR+4D Radar Fusion Detector<a href="#table-of-contents">🔝</a> 


### *2024*


* **Towards Robust 3D Object Detection with LiDAR and 4D Radar Fusion in Various Weather Conditions**, CVPR 2024                  
[[paper](https://openaccess.thecvf.com/content/CVPR2024/html/Chae_Towards_Robust_3D_Object_Detection_with_LiDAR_and_4D_Radar_CVPR_2024_paper.html)] [[code](https://github.com/yujeong-star/RL_3DOD)]  

 * **LiDAR-based All-weather 3D Object Detection via Prompting and Distilling 4D Radar**, ECCV 2024                  
[[paper](https://link.springer.com/chapter/10.1007/978-3-031-72992-8_21)] [[code](https://github.com/yujeong-star/LOD_PDR)]  
### *2025*

* **L4DR: LiDAR-4DRadar Fusion for Weather-Robust 3D Object Detection**, AAAI 2025                  
[[paper](https://arxiv.org/pdf/2408.03677)] [[code](https://github.com/ylwhxht/L4DR)]  


---
## <a id="icp"> with Cooperative Perception <a href="#table-of-contents">🔝</a> 
### *2024*

* **V2X-DGW: Domain Generalization for Multi-agent Perception under Adverse Weather Conditions**, Arxiv 2024                  
[[paper](https://arxiv.org/html/2403.11371)]    

 * **Weather-Aware Collaborative Perception With Uncertainty Reduction has been published**, TITS 2024                  
[[paper](https://ieeexplore.ieee.org/document/10739668)] [[data](https://waymo.com/open/challenges/2020/domain-adaptation/)]   
### *2025*

 * **V2X-R: Cooperative LiDAR-4D Radar Fusion for 3D Object Detection with Denoising Diffusion**, CVPR 2025                  
[[paper](https://arxiv.org/abs/2411.08402)] [[code](https://github.com/ylwhxht/V2X-R)]      
