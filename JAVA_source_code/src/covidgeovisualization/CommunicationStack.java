package covidgeovisualization;

import java.util.LinkedList;
import java.util.List;

public class CommunicationStack {

    private List<String> internalList = new LinkedList<>();

    public void push(String element) {
        this.internalList.add(0, element);
    }

    public String pop() {
        return this.internalList.remove(0);
    }

    public List<String> getInternalList() {
        return this.internalList;
    }

    public void pushAll(List<String> elements) {
        for (String element : elements) {
            push(element);
        }
    }
}

/* Location:              F:\MySoftwareDevelops\COVID\bucket-renormalization\seattle\COVIDGeoVisualization.jar!\covidgeovisualization\CommunicationStack.class
 * Java compiler version: 8 (52.0)
 * JD-Core Version:       1.1.3
 */
