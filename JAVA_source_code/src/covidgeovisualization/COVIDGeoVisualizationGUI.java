/*    */ package covidgeovisualization;
/*    */ 
///*    */ import de.milchreis.uibooster.UiBooster;
/*    */ import processing.core.PApplet;
/*    */ 
/*    */ public class COVIDGeoVisualizationGUI
/*    */   extends PApplet
/*    */ {
/*    */   COVIDGeoVisualization myParent;
/*    */   
/*    */   COVIDGeoVisualizationGUI(COVIDGeoVisualization parent) {
/* 20 */     this.myParent = parent;
/*    */   }
/*    */   
/*    */   public void setup() {
/* 24 */     size(800, 400, "processing.core.PGraphicsJava2D");
/* 25 */     background(10);
/*    */     
/* 27 */     rectMode(3);
/* 28 */     textAlign(3, 3);
///* 29 */     UiBooster booster = new UiBooster();
///* 30 */     booster.createForm("UI").addButton("Show/Hide tract index", new Runnable()
///*    */         {
///*    */           public void run() {
///* 33 */             if (!COVIDGeoVisualizationGUI.this.myParent.isShowTractIndex) {
///* 34 */               COVIDGeoVisualizationGUI.this.myParent.isShowTractIndex = true;
///* 35 */             } else if (COVIDGeoVisualizationGUI.this.myParent.isShowTractIndex == true) {
///* 36 */               COVIDGeoVisualizationGUI.this.myParent.isShowTractIndex = false;
///*    */             } 
///*    */           }
///* 39 */         }).run();
/*    */   }
/*    */ }


/* Location:              F:\MySoftwareDevelops\COVID\bucket-renormalization\seattle\COVIDGeoVisualization.jar!\covidgeovisualization\COVIDGeoVisualizationGUI.class
 * Java compiler version: 8 (52.0)
 * JD-Core Version:       1.1.3
 */