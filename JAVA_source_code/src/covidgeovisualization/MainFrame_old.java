/*     */ package covidgeovisualization;
/*     */ 
/*     */ import de.fhpotsdam.unfolding.geo.Location;
/*     */ import java.awt.event.ActionEvent;
/*     */ import java.awt.event.ActionListener;
/*     */ import javax.swing.BorderFactory;
/*     */ import javax.swing.GroupLayout;
/*     */ import javax.swing.JButton;
/*     */ import javax.swing.JFrame;
/*     */ import javax.swing.JLabel;
/*     */ import javax.swing.JList;
/*     */ import javax.swing.JPanel;
/*     */ import javax.swing.JScrollPane;
/*     */ import javax.swing.JSlider;
/*     */ import javax.swing.JToggleButton;
/*     */ import javax.swing.LayoutStyle;
/*     */ import javax.swing.event.ChangeEvent;
/*     */ import javax.swing.event.ChangeListener;
/*     */ import javax.swing.event.ListSelectionEvent;
/*     */ import javax.swing.event.ListSelectionListener;
/*     */ import py4j.GatewayServer;
/*     */ 
/*     */ 
/*     */ 
/*     */ public class MainFrame_old
/*     */   extends JFrame
/*     */ {
/*     */   public COVIDGeoVisualization child;
/*     */   public boolean isSelectionNeededConnectedExternally = false;
/*     */   public Py4jRPC rPC;
/*  31 */   public int stepSize = 0; private JButton jButton1; private JButton jButton2; private JButton jButton3; private JButton jButton4; private JLabel jLabel1;
/*     */   private JLabel jLabel2;
/*     */   private JLabel jLabel3;
/*     */   private JLabel jLabel4;
/*     */   public JList<String> jList1;
/*     */   /*     */   
/*     */   public MainFrame_old() {
/*  38 */     initComponents();
/*  39 */     runRPC();
/*     */   }
/*     */   private JPanel jPanel1; private JScrollPane jScrollPane1; private JSlider jSlider1; public JToggleButton jToggleButton1; public JToggleButton jToggleButton2; public JToggleButton jToggleButton3; public JToggleButton jToggleButton4; private JToggleButton jToggleButton5; private JToggleButton jToggleButton6; public JToggleButton jToggleButton7;
/*     */   public void runRPC() {
///*  43 */     this.rPC = new Py4jRPC(this);
/*  44 */     GatewayServer gatewayServer = new GatewayServer(this.rPC);
/*  45 */     gatewayServer.start();
/*  46 */     System.out.println("Gateway Server Started");
/*     */   }
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */   /*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */ 
/*     */   
/*     */   private void initComponents() {
/*  58 */     this.jButton1 = new JButton();
/*  59 */     this.jToggleButton1 = new JToggleButton();
/*  60 */     this.jToggleButton2 = new JToggleButton();
/*  61 */     this.jToggleButton3 = new JToggleButton();
/*  62 */     this.jToggleButton4 = new JToggleButton();
/*  63 */     this.jToggleButton5 = new JToggleButton();
/*  64 */     this.jLabel1 = new JLabel();
/*  65 */     this.jLabel2 = new JLabel();
/*  66 */     this.jLabel3 = new JLabel();
/*  67 */     this.jButton2 = new JButton();
/*  68 */     this.jButton3 = new JButton();
/*  69 */     this.jToggleButton6 = new JToggleButton();
/*  70 */     this.jPanel1 = new JPanel();
/*  71 */     this.jScrollPane1 = new JScrollPane();
/*  72 */     this.jList1 = new JList<>();
/*  73 */     this.jToggleButton7 = new JToggleButton();
/*  74 */     this.jSlider1 = new JSlider();
/*  75 */     this.jLabel4 = new JLabel();
/*  76 */     this.jButton4 = new JButton();
/*     */     
/*  78 */     setDefaultCloseOperation(3);
/*     */     
/*  80 */     this.jButton1.setText("Save image");
/*  81 */     this.jButton1.addActionListener(new ActionListener() {
/*     */           public void actionPerformed(ActionEvent evt) {
/*  83 */             MainFrame_old.this.jButton1ActionPerformed(evt);
/*     */           }
/*     */         });
/*     */     
/*  87 */     this.jToggleButton1.setText("Toggle tract index");
/*  88 */     this.jToggleButton1.setEnabled(false);
/*  89 */     this.jToggleButton1.addActionListener(new ActionListener() {
/*     */           public void actionPerformed(ActionEvent evt) {
/*  91 */             MainFrame_old.this.jToggleButton1ActionPerformed(evt);
/*     */           }
/*     */         });
/*     */     
/*  95 */     this.jToggleButton2.setText("Toggle undirected flow");
/*  96 */     this.jToggleButton2.setEnabled(false);
/*  97 */     this.jToggleButton2.addActionListener(new ActionListener() {
/*     */           public void actionPerformed(ActionEvent evt) {
/*  99 */             MainFrame_old.this.jToggleButton2ActionPerformed(evt);
/*     */           }
/*     */         });
/*     */     
/* 103 */     this.jToggleButton3.setText("Population");
/* 104 */     this.jToggleButton3.setEnabled(false);
/* 105 */     this.jToggleButton3.addActionListener(new ActionListener() {
/*     */           public void actionPerformed(ActionEvent evt) {
/* 107 */             MainFrame_old.this.jToggleButton3ActionPerformed(evt);
/*     */           }
/*     */         });
/*     */     
/* 111 */     this.jToggleButton4.setText("Population Density");
/* 112 */     this.jToggleButton4.setEnabled(false);
/* 113 */     this.jToggleButton4.addActionListener(new ActionListener() {
/*     */           public void actionPerformed(ActionEvent evt) {
/* 115 */             MainFrame_old.this.jToggleButton4ActionPerformed(evt);
/*     */           }
/*     */         });
/*     */     
/* 119 */     this.jToggleButton5.setSelected(true);
/* 120 */     this.jToggleButton5.setText("Show numbers on map");
/* 121 */     this.jToggleButton5.addActionListener(new ActionListener() {
/*     */           public void actionPerformed(ActionEvent evt) {
/* 123 */             MainFrame_old.this.jToggleButton5ActionPerformed(evt);
/*     */           }
/*     */         });
/*     */     
/* 127 */     this.jLabel1.setText("Not connected to Python");
/*     */     
/* 129 */     this.jLabel2.setText("Step number:");
/*     */     
/* 131 */     this.jLabel3.setText("NA");
/*     */     
/* 133 */     this.jButton2.setText("Next step");
/* 134 */     this.jButton2.setEnabled(false);
/* 135 */     this.jButton2.addActionListener(new ActionListener() {
/*     */           public void actionPerformed(ActionEvent evt) {
/* 137 */             MainFrame_old.this.jButton2ActionPerformed(evt);
/*     */           }
/*     */         });
/*     */     
/* 141 */     this.jButton3.setText("Finish selecting initial infection");
/* 142 */     this.jButton3.setEnabled(false);
/* 143 */     this.jButton3.addActionListener(new ActionListener() {
/*     */           public void actionPerformed(ActionEvent evt) {
/* 145 */             MainFrame_old.this.jButton3ActionPerformed(evt);
/*     */           }
/*     */         });
/*     */     
/* 149 */     this.jToggleButton6.setText("Infected graph");
/* 150 */     this.jToggleButton6.setEnabled(false);
/* 151 */     this.jToggleButton6.addActionListener(new ActionListener() {
/*     */           public void actionPerformed(ActionEvent evt) {
/* 153 */             MainFrame_old.this.jToggleButton6ActionPerformed(evt);
/*     */           }
/*     */         });
/*     */     
/* 157 */     this.jPanel1.setBorder(BorderFactory.createTitledBorder("Available map providers"));
/*     */     
/* 159 */     this.jList1.setSelectionMode(0);
/* 160 */     this.jList1.addListSelectionListener(new ListSelectionListener() {
/*     */           public void valueChanged(ListSelectionEvent evt) {
/* 162 */             MainFrame_old.this.jList1ValueChanged(evt);
/*     */           }
/*     */         });
/* 165 */     this.jScrollPane1.setViewportView(this.jList1);
/*     */     
/* 167 */     GroupLayout jPanel1Layout = new GroupLayout(this.jPanel1);
/* 168 */     this.jPanel1.setLayout(jPanel1Layout);
/* 169 */     jPanel1Layout.setHorizontalGroup(jPanel1Layout
/* 170 */         .createParallelGroup(GroupLayout.Alignment.LEADING)
/* 171 */         .addGroup(jPanel1Layout.createSequentialGroup()
/* 172 */           .addContainerGap()
/* 173 */           .addComponent(this.jScrollPane1, -1, 337, 32767)
/* 174 */           .addContainerGap()));
/*     */     
/* 176 */     jPanel1Layout.setVerticalGroup(jPanel1Layout
/* 177 */         .createParallelGroup(GroupLayout.Alignment.LEADING)
/* 178 */         .addGroup(jPanel1Layout.createSequentialGroup()
/* 179 */           .addContainerGap()
/* 180 */           .addComponent(this.jScrollPane1)
/* 181 */           .addContainerGap()));
/*     */ 
/*     */     
/* 184 */     this.jToggleButton7.setText("Adjusted probability of tracts");
/* 185 */     this.jToggleButton7.setEnabled(false);
/* 186 */     this.jToggleButton7.addActionListener(new ActionListener() {
/*     */           public void actionPerformed(ActionEvent evt) {
/* 188 */             MainFrame_old.this.jToggleButton7ActionPerformed(evt);
/*     */           }
/*     */         });
/*     */     /*     */     
/* 192 */     this.jSlider1.addChangeListener(new ChangeListener() {
/*     */           public void stateChanged(ChangeEvent evt) {
/* 194 */             MainFrame_old.this.jSlider1StateChanged(evt);
/*     */           }
/*     */         });
/*     */     
/* 198 */     this.jLabel4.setText("Zoom:");
/*     */     
/* 200 */     this.jButton4.setText("Go to Seattle");
/* 201 */     this.jButton4.addActionListener(new ActionListener() {
/*     */           public void actionPerformed(ActionEvent evt) {
/* 203 */             MainFrame_old.this.jButton4ActionPerformed(evt);
/*     */           }
/*     */         });
/*     */     
/* 207 */     GroupLayout layout = new GroupLayout(getContentPane());
/* 208 */     getContentPane().setLayout(layout);
/* 209 */     layout.setHorizontalGroup(layout
/* 210 */         .createParallelGroup(GroupLayout.Alignment.LEADING)
/* 211 */         .addGroup(layout.createSequentialGroup()
/* 212 */           .addContainerGap()
/* 213 */           .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
/* 214 */             .addGroup(layout.createSequentialGroup()
/* 215 */               .addComponent(this.jButton1)
/* 216 */               .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED, -1, 32767)
/* 217 */               .addComponent(this.jButton4)
/* 218 */               .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
/* 219 */               .addComponent(this.jToggleButton5))
/* 220 */             .addGroup(layout.createSequentialGroup()
/* 221 */               .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
/* 222 */                 .addComponent(this.jLabel1)
/* 223 */                 .addGroup(layout.createSequentialGroup()
/* 224 */                   .addComponent(this.jLabel2)
/* 225 */                   .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
/* 226 */                   .addComponent(this.jLabel3))
/* 227 */                 .addComponent(this.jButton2)
/* 228 */                 .addComponent(this.jButton3)
/* 229 */                 .addComponent(this.jToggleButton6)
/* 230 */                 .addComponent(this.jToggleButton3)
/* 231 */                 .addComponent(this.jToggleButton1)
/* 232 */                 .addComponent(this.jToggleButton2)
/* 233 */                 .addComponent(this.jToggleButton4)
/* 234 */                 .addComponent(this.jToggleButton7))
/* 235 */               .addGap(18, 18, 18)
/* 236 */               .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
/* 237 */                 .addGroup(layout.createSequentialGroup()
/* 238 */                   .addComponent(this.jLabel4)
/* 239 */                   .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
/* 240 */                   .addComponent(this.jSlider1, -1, -1, 32767))
/* 241 */                 .addComponent(this.jPanel1, -1, -1, 32767))))
/* 242 */           .addContainerGap()));
/*     */     
/* 244 */     layout.setVerticalGroup(layout
/* 245 */         .createParallelGroup(GroupLayout.Alignment.LEADING)
/* 246 */         .addGroup(GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
/* 247 */           .addContainerGap()
/* 248 */           .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING)
/* 249 */             .addGroup(layout.createSequentialGroup()
/* 250 */               .addComponent(this.jToggleButton1)
/* 251 */               .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
/* 252 */               .addComponent(this.jToggleButton2)
/* 253 */               .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
/* 254 */               .addComponent(this.jToggleButton3)
/* 255 */               .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
/* 256 */               .addComponent(this.jToggleButton4)
/* 257 */               .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
/* 258 */               .addComponent(this.jToggleButton7)
/* 259 */               .addGap(18, 18, 18)
/* 260 */               .addComponent(this.jToggleButton6)
/* 261 */               .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED, 103, 32767)
/* 262 */               .addComponent(this.jButton3)
/* 263 */               .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
/* 264 */               .addComponent(this.jButton2)
/* 265 */               .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
/* 266 */               .addGroup(layout.createParallelGroup(GroupLayout.Alignment.BASELINE)
/* 267 */                 .addComponent(this.jLabel2)
/* 268 */                 .addComponent(this.jLabel3))
/* 269 */               .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
/* 270 */               .addComponent(this.jLabel1))
/* 271 */             .addGroup(layout.createSequentialGroup()
/* 272 */               .addComponent(this.jPanel1, -1, -1, 32767)
/* 273 */               .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
/* 274 */               .addGroup(layout.createParallelGroup(GroupLayout.Alignment.LEADING, false)
/* 275 */                 .addComponent(this.jLabel4, -1, -1, 32767)
/* 276 */                 .addComponent(this.jSlider1, -2, 0, 32767))
/* 277 */               .addGap(12, 12, 12)))
/* 278 */           .addPreferredGap(LayoutStyle.ComponentPlacement.RELATED)
/* 279 */           .addGroup(layout.createParallelGroup(GroupLayout.Alignment.BASELINE)
/* 280 */             .addComponent(this.jButton1)
/* 281 */             .addComponent(this.jToggleButton5)
/* 282 */             .addComponent(this.jButton4))
/* 283 */           .addContainerGap()));
/*     */ 
/*     */     
/* 286 */     pack();
/*     */   }
/*     */   
/*     */   private void jButton1ActionPerformed(ActionEvent evt) {
/* 290 */     this.child.saveFile();
/*     */   }
/*     */   
/*     */   private void jToggleButton1ActionPerformed(ActionEvent evt) {
/* 294 */     if (!this.child.isShowTractIndex) {
/* 295 */       this.child.isShowTractIndex = true;
/* 296 */     } else if (this.child.isShowTractIndex == true) {
/* 297 */       this.child.isShowTractIndex = false;
/*     */     } 
/*     */   }
/*     */   
/*     */   private void jToggleButton2ActionPerformed(ActionEvent evt) {
/* 302 */     if (!this.child.isShowUndirectedFlow) {
/* 303 */       this.child.isShowUndirectedFlow = true;
/* 304 */     } else if (this.child.isShowUndirectedFlow == true) {
/* 305 */       this.child.isShowUndirectedFlow = false;
/*     */     } 
/*     */   }
/*     */   
/*     */   private void jToggleButton3ActionPerformed(ActionEvent evt) {
/* 310 */     if (!this.child.isShowPopulation) {
/* 311 */       this.child.isShowPopulation = true;
/* 312 */     } else if (this.child.isShowPopulation == true) {
/* 313 */       this.child.isShowPopulation = false;
/*     */     } 
/*     */   }
/*     */   
/*     */   private void jToggleButton4ActionPerformed(ActionEvent evt) {
/* 318 */     if (!this.child.isShowDensity) {
/* 319 */       this.child.isShowDensity = true;
/* 320 */     } else if (this.child.isShowDensity == true) {
/* 321 */       this.child.isShowDensity = false;
/*     */     } 
/*     */   }
/*     */   
/*     */   private void jToggleButton5ActionPerformed(ActionEvent evt) {
/* 326 */     if (!this.child.isShowNumber) {
/* 327 */       this.child.isShowNumber = true;
/* 328 */     } else if (this.child.isShowNumber == true) {
/* 329 */       this.child.isShowNumber = false;
/*     */     } 
/*     */   }
/*     */   
/*     */   private void jButton3ActionPerformed(ActionEvent evt) {
/* 334 */     for (int i = 0; i < this.child.selectedInitialInfections.length; i++) {
/* 335 */       if (this.child.selectedInitialInfections[i] == true) {
/* 336 */         this.rPC.getSelectionStack().push("1");
/*     */       } else {
/* 338 */         this.rPC.getSelectionStack().push("0");
/*     */       } 
/*     */     } 
/* 341 */     this.isSelectionNeededConnectedExternally = false;
/* 342 */     this.child.isShowTractIndex = false;
/* 343 */     this.jToggleButton1.setSelected(false);
/* 344 */     this.rPC.getRefreshStack().pop();
/* 345 */     this.jButton2.setEnabled(true);
/* 346 */     this.child.isShowInfectedGraph = true;
/* 347 */     this.jToggleButton6.setSelected(true);
/* 348 */     this.jButton3.setEnabled(false);
/* 349 */     this.stepSize = 0;
/*     */   }
/*     */   
/*     */   private void jButton2ActionPerformed(ActionEvent evt) {
/* 353 */     if (this.rPC.getRefreshStack().getInternalList().size() > 0) {
/* 354 */       this.rPC.getRefreshStack().pop();
/* 355 */       this.stepSize++;
/* 356 */       this.jLabel3.setText(String.valueOf(this.stepSize));
/*     */     } 
/*     */   }
/*     */   
/*     */   private void jToggleButton6ActionPerformed(ActionEvent evt) {
/* 361 */     if (!this.child.isShowInfectedGraph) {
/* 362 */       this.child.isShowInfectedGraph = true;
/* 363 */     } else if (this.child.isShowInfectedGraph == true) {
/* 364 */       this.child.isShowInfectedGraph = false;
/*     */     } 
/*     */   }
/*     */   
/*     */   private void jList1ValueChanged(ListSelectionEvent evt) {
/* 369 */     if (this.child != null && 
/* 370 */       this.child.mapSources != null && 
/* 371 */       ((MapSourse)this.child.mapSources.maps.get(this.jList1.getSelectedIndex())).map != null) {
/* 372 */       this.child.saveLastLocation();
/* 373 */       this.child.map = ((MapSourse)this.child.mapSources.maps.get(this.jList1.getSelectedIndex())).map;
///* 374 */       this.child.resetMap();
/* 375 */       this.child.zoomLevel = 12.0F;
/* 376 */       this.jSlider1.setValue(62);
/*     */     } 
/*     */   }
/*     */ 
/*     */ 
/*     */   
/*     */   private void jToggleButton7ActionPerformed(ActionEvent evt) {
/* 383 */     if (!this.child.isShowNodeAdjustedProbability) {
/* 384 */       this.child.isShowNodeAdjustedProbability = true;
/* 385 */     } else if (this.child.isShowNodeAdjustedProbability == true) {
/* 386 */       this.child.isShowNodeAdjustedProbability = false;
/*     */     } 
/*     */   }
/*     */   
/*     */   private void jSlider1StateChanged(ChangeEvent evt) {
/* 391 */     this.child.zoomLevel = 2.0F + this.jSlider1.getValue() / 100.0F * 16.0F;
/* 392 */     this.child.saveLastLocation();
///* 393 */     this.child.resetMap();
/*     */   }
/*     */   
/*     */   private void jButton4ActionPerformed(ActionEvent evt) {
/* 397 */     this.child.map.panTo(new Location(47.618923F, -122.33458F));
/*     */   }
/*     */   
/*     */   public void connectionFound() {
/* 401 */     this.jLabel1.setText("Connected to Python");
/* 402 */     this.rPC.getRefreshStack().push("1");
/* 403 */     this.child.initSelection();
/* 404 */     this.isSelectionNeededConnectedExternally = true;
/* 405 */     this.child.isShowTractIndex = true;
/* 406 */     this.child.isShowNumber = true;
/* 407 */     this.jToggleButton1.setSelected(true);
/* 408 */     this.jToggleButton6.setEnabled(true);
/* 409 */     this.jButton3.setEnabled(true);
/* 410 */     this.stepSize = 0;
/* 411 */     this.jLabel3.setText(String.valueOf(this.stepSize));
/*     */   }
/*     */ }


/* Location:              F:\MySoftwareDevelops\COVID\bucket-renormalization\seattle\COVIDGeoVisualization.jar!\covidgeovisualization\MainFrame.class
 * Java compiler version: 8 (52.0)
 * JD-Core Version:       1.1.3
 */