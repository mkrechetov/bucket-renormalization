package covidgeovisualization;

import de.fhpotsdam.unfolding.UnfoldingMap;
import de.fhpotsdam.unfolding.geo.Location;
import de.fhpotsdam.unfolding.marker.SimplePointMarker;
import de.fhpotsdam.unfolding.utils.MapUtils;
import de.fhpotsdam.unfolding.utils.ScreenPosition;
import de.siegmar.fastcsv.reader.CsvContainer;
import de.siegmar.fastcsv.reader.CsvReader;
import de.siegmar.fastcsv.reader.CsvRow;
import de.siegmar.fastcsv.writer.CsvWriter;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.AbstractListModel;
import processing.core.PApplet;
import processing.core.PFont;
import processing.core.PVector;

public class COVIDGeoVisualization
        extends PApplet {

    UnfoldingMap map;
    float[][] rawFlowData;
    Location[] latLons;
    ScreenPosition[] uvCoordinates;
    float[][] thirdPointDeviationPercentages;
    float[] population;
    float[] populationNormalized;
    float[] density;
    float[] densityNormalized;
    float[] adjustedProbability;
    float[] adjustedProbabilityNormalized;
    int[] status;
    boolean[] selectedInitialInfections;
    MainFrame parent;
    boolean isSave = false;
    int selectedMapIndex = 0;

    boolean isShowTractIndex = false;

    boolean isShowUndirectedFlow = false;
    boolean isShowPopulation = false;
    boolean isShowDensity = false;
    boolean isShowInfectedGraph = false;
    boolean isShowNodeAdjustedProbability = false;
    boolean isShowNumber = true;
    boolean isChangeCaseStudy = false;
    boolean isReadyChangeCaseStudy = false;
    public float zoomLevel;

    public Location lastLocation;
    MapSources mapSources;
    ArrayList<CaseStudy> caseStudies = new ArrayList();

    COVIDGeoVisualization() {
    }

    COVIDGeoVisualization(MainFrame mainFrame) {
        this.parent = mainFrame;
    }

    @Override
    public void settings() {
        size(1000, 1000, "processing.opengl.PGraphics3D");
    }

    @Override
    public void setup() {
//        size(500, 500, "processing.opengl.PGraphics3D");

        this.mapSources = new MapSources(this);
        final String[] mapNames = new String[this.mapSources.maps.size()];
        for (int i = 0; i < this.mapSources.maps.size(); i++) {
            mapNames[i] = ((MapSourse) this.mapSources.maps.get(i)).name;
        }
        this.parent.jList1.setModel(new AbstractListModel<String>() {
            @Override
            public int getSize() {
                return mapNames.length;
            }

            @Override
            public String getElementAt(int index) {
                return mapNames[index];
            }
        });
        for (int i = 0; i < this.mapSources.maps.size(); i++) {
            if (((MapSourse) this.mapSources.maps.get(i)).map != null) {
                this.map = ((MapSourse) this.mapSources.maps.get(i)).map;
                this.parent.jList1.setSelectedIndex(i);

                break;
            }
        }

        PFont font = createFont("serif-bold", 14.0F);
        textFont(font);
        textMode(4);

        this.parent.child = this;
        setupCaseStudies();
        readCaseStudyData(0);
        resetMap(0);
        setUVCoordinates(0);
    }

    public void setUVCoordinates(int caseStudyIndex) {
        try {
            this.uvCoordinates = new ScreenPosition[this.latLons.length];
            ArrayList<String[]> rows = (ArrayList) new ArrayList<>();
            String[] row = new String[3];
            row[0] = String.valueOf("Tract");
            row[1] = String.valueOf("UVX");
            row[2] = String.valueOf("UVY");
            rows.add(row);
            for (int k = 0; k < this.latLons.length; k++) {
                SimplePointMarker locSM = new SimplePointMarker(this.latLons[k]);
                ScreenPosition scLocPos = null;
                try {
                    scLocPos = locSM.getScreenPosition(this.map);
                } catch (Exception ex) {
                    System.out.println("LLL");
                }

                this.uvCoordinates[k] = scLocPos;
                row = new String[3];
                row[0] = String.valueOf(k);
                row[1] = String.valueOf(scLocPos.x);
                row[2] = String.valueOf(scLocPos.y);
                rows.add(row);
            }
            CsvWriter writer = new CsvWriter();
            writer.write(new File(this.caseStudies.get(caseStudyIndex).name + "_UV_coordinates.csv"), StandardCharsets.UTF_8, rows);
        } catch (IOException ex) {
            Logger.getLogger(COVIDGeoVisualization.class.getName()).log(Level.SEVERE, (String) null, ex);
        }
    }

    public void readCaseStudyData(int caseStudyIndex) {
        String GISDataFileName = this.caseStudies.get(caseStudyIndex).name + "_GIS_data.csv";
        File GISDataFile = new File(GISDataFileName);
        try {
            CsvReader cSVReader = new CsvReader();
            cSVReader.setContainsHeader(true);
            CsvContainer data = cSVReader.read(GISDataFile, StandardCharsets.UTF_8);
            this.latLons = new Location[data.getRows().size()];
            this.status = new int[this.latLons.length];
            this.population = new float[data.getRows().size()];
            this.density = new float[data.getRows().size()];
            this.adjustedProbability = new float[latLons.length];//NEED TO BE LOADED LATER
            for (int j = 0; j < data.getRows().size(); j++) {
                try {
                    this.latLons[j] = new Location(Float.parseFloat(((CsvRow) data.getRows().get(j)).getFields().get(1)), Float.parseFloat(((CsvRow) data.getRows().get(j)).getFields().get(2)));
                } catch (Exception ex) {

                }
                try {
                    this.population[j] = Float.parseFloat((String) ((CsvRow) data.getRows().get(j)).getFields().get(3));
                } catch (Exception ex) {

                }
                try {
                    this.density[j] = Float.parseFloat((String) ((CsvRow) data.getRows().get(j)).getFields().get(4));
                } catch (Exception ex) {

                }
            }
            this.parent.jToggleButton1.setEnabled(true);

            this.populationNormalized = normalizeVector(this.population);
            this.densityNormalized = normalizeVector(this.density);
            this.parent.jToggleButton3.setEnabled(true);
            this.parent.jToggleButton4.setEnabled(true);
        } catch (IOException ex) {
            Logger.getLogger(COVIDGeoVisualization.class.getName()).log(Level.SEVERE, (String) null, ex);
        }

        String travelProbabilitiesFileName = this.caseStudies.get(caseStudyIndex).name + "_travel_probabilities.csv";
        float maxThirdPointCurveDeviationPercent = 2.0F;
        float minThirdPointCurveDeviationPercent = -2.0F;
        File fileFlow = new File(travelProbabilitiesFileName);
        try {
            CsvReader cSVReader = new CsvReader();
            cSVReader.setContainsHeader(false);
            CsvContainer data = cSVReader.read(fileFlow, StandardCharsets.UTF_8);
            this.rawFlowData = new float[data.getRows().size()][((CsvRow) data.getRows().get(0)).getFields().size()];
            this.thirdPointDeviationPercentages = new float[data.getRows().size()][((CsvRow) data.getRows().get(0)).getFields().size()];
            for (int j = 0; j < data.getRows().size(); j++) {
                for (int k = j + 1; k < ((CsvRow) data.getRows().get(j)).getFields().size(); k++) {
                    this.rawFlowData[j][k] = Float.parseFloat((String) ((CsvRow) data.getRows().get(j)).getFields().get(k));
                    this.thirdPointDeviationPercentages[j][k] = (float) (minThirdPointCurveDeviationPercent + Math.random() * (maxThirdPointCurveDeviationPercent - minThirdPointCurveDeviationPercent));
                }
            }
            this.rawFlowData = normalizeMatrix2D(this.rawFlowData);
            this.parent.jToggleButton2.setEnabled(true);
        } catch (IOException ex) {
            Logger.getLogger(COVIDGeoVisualization.class.getName()).log(Level.SEVERE, (String) null, ex);
        }
    }

    public void setupCaseStudies() {
        caseStudies.add(new CaseStudy("seattle", new Location(47.618923F, -122.33458F), 12f));
        caseStudies.add(new CaseStudy("wisconsin", new Location(44.806469F, -89.680672F), 7.7f));
        final String[] caseStudyNames = new String[this.caseStudies.size()];
        int i;
        for (i = 0; i < this.caseStudies.size(); i++) {
            caseStudyNames[i] = ((CaseStudy) this.caseStudies.get(i)).name;
        }
        this.parent.jList2.setModel(new AbstractListModel<String>() {
            @Override
            public int getSize() {
                return caseStudyNames.length;
            }

            @Override
            public String getElementAt(int index) {
                return caseStudyNames[index];
            }
        });
        this.parent.jList2.setSelectedIndex(0);
//        int value=2.0F + this.jSlider1.getValue() / 100.0F * 16.0F;

    }

    public void resetMap(int caseStudyIndex) {
        zoomLevel = this.caseStudies.get(caseStudyIndex).zoomLevel;
        this.map.zoomTo(this.caseStudies.get(caseStudyIndex).zoomLevel);
        this.map.panTo(this.caseStudies.get(caseStudyIndex).mapLatLon);
        int value = (int) (((this.caseStudies.get(caseStudyIndex).zoomLevel - 2.0) / 16f) * 100f);
        this.parent.jSlider1.setValue(value);
        MapUtils.createDefaultEventDispatcher(this, new UnfoldingMap[]{this.map});
    }

    public void saveLastLocation() {
        this.lastLocation = this.map.getCenter();
    }

    public void resetMap() {
        this.map.zoomTo(this.zoomLevel);
        this.map.panTo(this.lastLocation);
        MapUtils.createDefaultEventDispatcher(this, new UnfoldingMap[]{this.map});
    }

    public void initSelection() {
        this.selectedInitialInfections = new boolean[this.latLons.length];
        for (int i = 0; i < this.latLons.length; i++) {
            this.selectedInitialInfections[i] = false;
        }
    }

    public float[][] normalizeMatrix2D(float[][] input) {
        float[][] output = new float[input.length][(input[0]).length];
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        int i;
        for (i = 0; i < input.length; i++) {
            for (int j = 0; j < (input[i]).length; j++) {
                if (input[i][j] > max) {
                    max = input[i][j];
                }
                if (input[i][j] < min) {
                    min = input[i][j];
                }
            }
        }
        for (i = 0; i < input.length; i++) {
            for (int j = 0; j < (input[i]).length; j++) {
                output[i][j] = (input[i][j] - min) / max;
            }
        }
        return output;
    }

    public float[] normalizeVector(float[] input) {
        float[] output = new float[input.length];
        float min = Float.MAX_VALUE;
        float max = Float.MIN_VALUE;
        int i;
        for (i = 0; i < input.length; i++) {
            if (input[i] > max) {
                max = input[i];
            }
            if (input[i] < min) {
                min = input[i];
            }
        }
        for (i = 0; i < input.length; i++) {
            output[i] = (input[i] - min) / max;
        }
        return output;
    }

    public void draw() {
        background(0);
        this.map.draw();

        if (isChangeCaseStudy == true) {
            isChangeCaseStudy = false;
            isReadyChangeCaseStudy = true;
        }
        if (isReadyChangeCaseStudy != true) {
//            System.out.println("Rendered!!!!!!");
            if (this.rawFlowData != null) {
                for (int i = 0; i < this.rawFlowData.length; i++) {
                    if (this.isShowTractIndex == true) {
                        drawIndexText(i, this.latLons[i]);
                    }
                    if (this.isShowPopulation == true) {
                        drawPopulationMarkers(i, this.latLons[i]);
                    }
                    if (this.isShowDensity == true) {
                        drawPopulationDensityMarkers(i, this.latLons[i]);
                    }
                    if (this.isShowInfectedGraph == true) {
                        drawInfectedGraph(i, this.latLons[i]);
                    }
                    if (this.parent.isSelectionNeededConnectedExternally == true) {
                        drawSelection(i, this.latLons[i]);
                    }
                    if (this.isShowNodeAdjustedProbability == true) {
                        drawAdjustedProbability(i, this.latLons[i]);
                    }
                }
            }
            if (this.isShowNodeAdjustedProbability == true) {
                fill(200.0F, 200.0F, 200.0F, 256.0F);
                rect(6.0F, 6.0F, 40.0F, 614.0F);
                for (int i = 0; i < 100; i++) {
                    fill((100 - i) / 100.0F * 256.0F, 0.0F, i / 100.0F * 256.0F, 256.0F);
                    rect(10.0F, (10 + 6 * i), 20.0F, 6.0F);
                }
                fill(0.0F, 0.0F, 0.0F);
                text("1", 34.0F, 20.0F);
                text("0", 34.0F, 610.0F);
            }
            if (this.isShowUndirectedFlow == true) {
                noFill();
                for (int i = 0; i < this.rawFlowData.length; i++) {
                    for (int j = i + 1; j < (this.rawFlowData[0]).length; j++) {
                        strokeWeight(2.0F + this.rawFlowData[i][j] * 4.0F);
                        stroke(40.0F + this.rawFlowData[i][j] * 220.0F, 0.0F, Math.max(0.0F, 230.0F - this.rawFlowData[i][j] * 200.0F), Math.max(0.0F, -48.0F + (float) Math.pow((this.rawFlowData[i][j] * 40000.0F), 0.48D)));
                        drawArc(i, j, this.latLons[i], this.latLons[j]);
                    }
                }
            }
            if (this.parent.rPC.isBusyRefreshing().equals("1") && this.parent.rPC.getGraphColorStack().getInternalList().size() > 0) {
                refreshInfected();
            }

            if (this.isSave == true) {
                saveFrame("output.jpg");
                this.isSave = false;
            }
        }
    }

    public void refreshInfected() {
        this.status = new int[this.latLons.length];
        int i;
        for (i = 0; i < this.latLons.length; i++) {
            if (((String) this.parent.rPC.getGraphColorStack().getInternalList().get(i)).equals("b")) {
                this.status[i] = 0;
            } else if (((String) this.parent.rPC.getGraphColorStack().getInternalList().get(i)).equals("r")) {
                this.status[i] = 1;
            } else if (((String) this.parent.rPC.getGraphColorStack().getInternalList().get(i)).equals("k")) {
                this.status[i] = 2;
            }
        }
        for (i = 0; i < this.latLons.length; i++) {
            this.parent.rPC.getGraphColorStack().pop();
        }
        this.parent.rPC.getRefreshStack().pop();
    }

    public void mouseClicked() {
        if (this.parent.isSelectionNeededConnectedExternally == true) {
            Location location = this.map.getLocation(this.mouseX, this.mouseY);
            float minDistance = Float.MAX_VALUE;
            int selectedIndex = -1;
            for (int i = 0; i < this.latLons.length; i++) {
                float dist = this.latLons[i].dist((PVector) location);
                if (dist < minDistance) {
                    selectedIndex = i;
                    minDistance = dist;
                }
            }
            if (this.selectedInitialInfections[selectedIndex] == true) {
                this.selectedInitialInfections[selectedIndex] = false;
            } else {
                this.selectedInitialInfections[selectedIndex] = true;
            }
        }
    }

    public void drawAdjustedProbability(int index, Location loc) {
        SimplePointMarker locSM = new SimplePointMarker(loc);
        ScreenPosition scLocPos = locSM.getScreenPosition(this.map);
        fill(this.adjustedProbabilityNormalized[index] * 256.0F, 0.0F, (1.0F - this.adjustedProbabilityNormalized[index]) * 256.0F, 256.0F);
        ellipse(scLocPos.x, scLocPos.y, 12.0F, 12.0F);
    }

    public void drawInfectedGraph(int index, Location loc) {
        SimplePointMarker locSM = new SimplePointMarker(loc);
        ScreenPosition scLocPos = locSM.getScreenPosition(this.map);
        if (this.status[index] == 0) {
            fill(0.0F, 0.0F, 256.0F, 256.0F);
        } else if (this.status[index] == 1) {
            fill(256.0F, 0.0F, 0.0F, 256.0F);
        } else if (this.status[index] == 2) {
            fill(0.0F, 0.0F, 0.0F, 256.0F);
        }
        ellipse(scLocPos.x, scLocPos.y, 15.0F, 15.0F);
    }

    public void drawSelection(int index, Location loc) {
        SimplePointMarker locSM = new SimplePointMarker(loc);
        ScreenPosition scLocPos = locSM.getScreenPosition(this.map);
        if (this.selectedInitialInfections[index] == true) {
            fill(0.0F, 0.0F, 200.0F, 100.0F);
        } else {
            fill(0.0F, 200.0F, 0.0F, 100.0F);
        }
        ellipse(scLocPos.x, scLocPos.y, 15.0F, 15.0F);
    }

    public void saveFile() {
        this.isSave = true;
    }

    public void drawPopulationMarkers(int index, Location loc) {
        SimplePointMarker locSM = new SimplePointMarker(loc);
        ScreenPosition scLocPos = locSM.getScreenPosition(this.map);
        fill(0.0F, 200.0F, 0.0F, 100.0F);
        ellipse(scLocPos.x, scLocPos.y, this.populationNormalized[index] * 40.0F, this.populationNormalized[index] * 40.0F);
        if (this.isShowNumber == true) {
            fill(200.0F, 0.0F, 0.0F, 256.0F);
            text((int) this.population[index], scLocPos.x - textWidth(String.valueOf((int) this.population[index])) / 2.0F, scLocPos.y + 4.0F);
        }
    }

    public void drawPopulationDensityMarkers(int index, Location loc) {
        SimplePointMarker locSM = new SimplePointMarker(loc);
        ScreenPosition scLocPos = locSM.getScreenPosition(this.map);
        fill(0.0F, 0.0F, 200.0F, 100.0F);
        ellipse(scLocPos.x, scLocPos.y, this.densityNormalized[index] * 40.0F, this.densityNormalized[index] * 40.0F);
        if (this.isShowNumber == true) {
            fill(200.0F, 0.0F, 0.0F, 256.0F);
            text((int) this.density[index], scLocPos.x - textWidth(String.valueOf((int) this.density[index])) / 2.0F, scLocPos.y + 4.0F);
        }
    }

    public void drawIndexText(int index, Location loc) {
        SimplePointMarker locSM = new SimplePointMarker(loc);
        ScreenPosition scLocPos = locSM.getScreenPosition(this.map);
        fill(0.0F, 200.0F, 0.0F, 100.0F);
        ellipse(scLocPos.x, scLocPos.y, 15.0F, 15.0F);
        if (this.isShowNumber == true) {
            fill(200.0F, 0.0F, 0.0F, 256.0F);
            text("Tract index: " + index, scLocPos.x - textWidth("Tract index: " + index) / 2.0F, scLocPos.y + 4.0F);
        }
    }

    public void drawArc(int sourceIndex, int destinationIndex, Location start, Location end) {
        float distance = start.dist((PVector) end);
        Location middlePoint = new Location(start.x + (end.x - start.x) / 2.0F, start.y + (end.y - start.y) / 2.0F);
        Location slope = new Location(end.x - start.x, end.y - start.y);
        Location inverseSlope = new Location(-1.0F / slope.x, -1.0F / slope.y);

        Location thirdPoint = new Location(middlePoint.x + distance * this.thirdPointDeviationPercentages[sourceIndex][destinationIndex] / 10.0F, middlePoint.y + distance * this.thirdPointDeviationPercentages[sourceIndex][destinationIndex] / 10.0F);

        SimplePointMarker startMarker = new SimplePointMarker(start);
        SimplePointMarker endMarker = new SimplePointMarker(end);
        SimplePointMarker middleMarker = new SimplePointMarker(thirdPoint);

        ScreenPosition scStartPos = startMarker.getScreenPosition(this.map);
        ScreenPosition scEndPos = endMarker.getScreenPosition(this.map);
        ScreenPosition scMiddlePos = middleMarker.getScreenPosition(this.map);

        beginShape();
        curveVertex(scStartPos.x, scStartPos.y);
        curveVertex(scStartPos.x, scStartPos.y);
        curveVertex(scMiddlePos.x, scMiddlePos.y);
        curveVertex(scEndPos.x, scEndPos.y);
        curveVertex(scEndPos.x, scEndPos.y);
        endShape();
    }

    public void startRendering() {
        String[] myArgs = {""};
        PApplet.runSketch(myArgs, this);
    }
}


/* Location:              F:\MySoftwareDevelops\COVID\bucket-renormalization\seattle\COVIDGeoVisualization.jar!\covidgeovisualization\COVIDGeoVisualization.class
 * Java compiler version: 8 (52.0)
 * JD-Core Version:       1.1.3
 */
