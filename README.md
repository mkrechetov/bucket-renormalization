# bucket renormalization
Forked from Sungsoo Ahn's repository to use his Global Bucket Renormalization (GBR) algorithm for approximate inference.
Contributors to this repo: Ruby and Amir
Goal: compute marginal probabilities that a census tract contracts the COVID-19 virus.  

* The main script of the code is "marginalization.py".  
* The "drawHeatMap.py" visualizes probabilities.  
* "runpy.pbs_script" runs different experiments with various parameters.  
* "./seattle/" contains the data related to Seattle case study and it has the compiled Java app for visualization.
* "./wisconsin/" contains the data related to Wisconsin counties.
