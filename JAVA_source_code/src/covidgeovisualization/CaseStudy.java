/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package covidgeovisualization;

import de.fhpotsdam.unfolding.geo.Location;

/**
 *
 * @author user
 */
public class CaseStudy {
    public String name;
    public Location mapLatLon;
    public float zoomLevel;
    
    public CaseStudy(String passed_name, Location passed_mapLatLon, float passed_zoom){
        name=passed_name;
        mapLatLon=passed_mapLatLon;
        zoomLevel=passed_zoom;
    }
}
