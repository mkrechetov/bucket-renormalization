package covidgeovisualization;

public class Py4jRPC {

    private CommunicationStack needRefreshStack;
    private CommunicationStack graphColorsStack;
    private CommunicationStack selectionStack;
    public boolean isConnected = false;
    MainFrame myParent;

    public Py4jRPC(MainFrame parent) {
        this.myParent = parent;
        this.needRefreshStack = new CommunicationStack();
        this.graphColorsStack = new CommunicationStack();
        this.selectionStack = new CommunicationStack();
    }

    public String isBusyRefreshing() {
        if (this.needRefreshStack.getInternalList().size() > 0) {
            return "1";
        }
        return "0";
    }

    public void connectToRPC() {
        this.isConnected = true;
        if (this.myParent != null) {
            this.myParent.connectionFound();
        }
    }

    public CommunicationStack getRefreshStack() {
        return this.needRefreshStack;
    }

    public CommunicationStack getGraphColorStack() {
        return this.graphColorsStack;
    }

    public CommunicationStack getSelectionStack() {
        return this.selectionStack;
    }
}

/* Location:              F:\MySoftwareDevelops\COVID\bucket-renormalization\seattle\COVIDGeoVisualization.jar!\covidgeovisualization\Py4jRPC.class
 * Java compiler version: 8 (52.0)
 * JD-Core Version:       1.1.3
 */
