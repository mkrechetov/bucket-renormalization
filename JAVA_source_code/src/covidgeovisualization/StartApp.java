package covidgeovisualization;

public class StartApp
{
public static void main(String[] args) {
MainFrame swingGui = new MainFrame();
swingGui.setVisible(true);

COVIDGeoVisualization sketch = new COVIDGeoVisualization(swingGui);
sketch.startRendering();
}
}

/* Location:              F:\MySoftwareDevelops\COVID\bucket-renormalization\seattle\COVIDGeoVisualization.jar!\covidgeovisualization\StartApp.class
 * Java compiler version: 8 (52.0)
 * JD-Core Version:       1.1.3
 */