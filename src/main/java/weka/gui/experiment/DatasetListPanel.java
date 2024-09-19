/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    DatasetListPanel.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.gui.experiment;

import java.awt.BorderLayout;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.File;
import java.util.Collections;
import java.util.Vector;

import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.filechooser.FileFilter;

import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.Saver;
import weka.experiment.Experiment;
import weka.gui.ConverterFileChooser;
import weka.gui.JListHelper;
import weka.gui.ViewerDialog;

/**
 * This panel controls setting a list of datasets for an experiment to iterate
 * over.
 * 
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @version $Revision: 10222 $
 */
public class DatasetListPanel extends JPanel implements ActionListener {

  /** for serialization. */
  private static final long serialVersionUID = 7068857852794405769L;

  /** The experiment to set the dataset list of. */
  protected Experiment m_Exp;

  /** The component displaying the dataset list. */
  protected JList m_List;

  /** Click to add a dataset. */
  protected JButton m_AddBut = new JButton("Add new...");

  /** Click to edit the selected algorithm. */
  protected JButton m_EditBut = new JButton("Edit selected...");

  /** Click to remove the selected dataset from the list. */
  protected JButton m_DeleteBut = new JButton("Delete selected");

  /** Click to move the selected dataset(s) one up. */
  protected JButton m_UpBut = new JButton("Up");

  /** Click to move the selected dataset(s) one down. */
  protected JButton m_DownBut = new JButton("Down");

  /** Make file paths relative to the user (start) directory. */
  protected JCheckBox m_relativeCheck = new JCheckBox("Use relative paths");

  /** The user (start) directory. */
  // protected File m_UserDir = new File(System.getProperty("user.dir"));

  /** The file chooser component. */
  protected ConverterFileChooser m_FileChooser = new ConverterFileChooser(
    ExperimenterDefaults.getInitialDatasetsDirectory());

  /** Indicates if the user has selected an experiment with training/test KEEL samples */
  protected boolean m_TraTstKEELSamples = false;

  /**
   * Creates the dataset list panel with the given experiment.
   * 
   * @param exp a value of type 'Experiment'
   */
  public DatasetListPanel(Experiment exp) {

    this();
    setExperiment(exp);
  }

  /**
   * Create the dataset list panel initially disabled.
   */
  public DatasetListPanel() {

    m_List = new JList();
    m_List.addListSelectionListener(new ListSelectionListener() {
      @Override
      public void valueChanged(ListSelectionEvent e) {
        setButtons(e);
      }
    });
    MouseListener mouseListener = new MouseAdapter() {
      @Override
      public void mouseClicked(MouseEvent e) {
        if (e.getClickCount() == 2) {
          // unfortunately, locationToIndex only returns the nearest entry
          // and not the exact one, i.e. if there's one item in the list and
          // one doublelclicks somewhere in the list, this index will be
          // returned
          int index = m_List.locationToIndex(e.getPoint());
          if (index > -1) {
            actionPerformed(new ActionEvent(m_EditBut, 0, ""));
          }
        }
      }
    };
    m_List.addMouseListener(mouseListener);

    MouseListener panelMouseListener = new MouseAdapter() {
    	public void mouseEntered(MouseEvent e) {
    		if (m_TraTstKEELSamples) {
        		m_AddBut.setText("Add main path ONLY!");
        		m_AddBut.setToolTipText("Add only the main path where the directories of the 3 data set contexts are located.");
        		m_relativeCheck.setEnabled(false);
        	    m_FileChooser.setMultiSelectionEnabled(false);
    		} else {
        		m_AddBut.setText("Add new...");
        		m_AddBut.setToolTipText("Add new...");
        		m_relativeCheck.setEnabled(true);
        	    m_FileChooser.setMultiSelectionEnabled(true);
    		}
    	}
    };
    addMouseListener(panelMouseListener);

    // m_FileChooser.setFileSelectionMode(JFileChooser.FILES_AND_DIRECTORIES);
    m_FileChooser.setCoreConvertersOnly(true);
    m_FileChooser.setMultiSelectionEnabled(true);
    m_FileChooser
      .setFileSelectionMode(ConverterFileChooser.FILES_AND_DIRECTORIES);
    m_FileChooser.setAcceptAllFileFilterUsed(false);
    m_DeleteBut.setEnabled(false);
    m_DeleteBut.addActionListener(this);
    m_AddBut.setEnabled(false);
    m_AddBut.addActionListener(this);
    m_EditBut.setEnabled(false);
    m_EditBut.addActionListener(this);
    m_UpBut.setEnabled(false);
    m_UpBut.addActionListener(this);
    m_DownBut.setEnabled(false);
    m_DownBut.addActionListener(this);
    m_relativeCheck.setSelected(ExperimenterDefaults.getUseRelativePaths());
    m_relativeCheck.setToolTipText("Store file paths relative to "
      + "the start directory");
    setLayout(new BorderLayout());
    setBorder(BorderFactory.createTitledBorder("Datasets"));
    JPanel topLab = new JPanel();
    GridBagLayout gb = new GridBagLayout();
    GridBagConstraints constraints = new GridBagConstraints();
    topLab.setBorder(BorderFactory.createEmptyBorder(10, 5, 10, 5));
    // topLab.setLayout(new GridLayout(1,2,5,5));
    topLab.setLayout(gb);

    constraints.gridx = 0;
    constraints.gridy = 0;
    constraints.weightx = 5;
    constraints.fill = GridBagConstraints.HORIZONTAL;
    constraints.gridwidth = 1;
    constraints.gridheight = 1;
    constraints.insets = new Insets(0, 2, 0, 2);
    topLab.add(m_AddBut, constraints);
    constraints.gridx = 1;
    constraints.gridy = 0;
    constraints.weightx = 5;
    constraints.gridwidth = 1;
    constraints.gridheight = 1;
    topLab.add(m_EditBut, constraints);
    constraints.gridx = 2;
    constraints.gridy = 0;
    constraints.weightx = 5;
    constraints.gridwidth = 1;
    constraints.gridheight = 1;
    topLab.add(m_DeleteBut, constraints);

    constraints.gridx = 0;
    constraints.gridy = 1;
    constraints.weightx = 5;
    constraints.fill = GridBagConstraints.HORIZONTAL;
    constraints.gridwidth = 1;
    constraints.gridheight = 1;
    constraints.insets = new Insets(0, 2, 0, 2);
    topLab.add(m_relativeCheck, constraints);

    JPanel bottomLab = new JPanel();
    gb = new GridBagLayout();
    constraints = new GridBagConstraints();
    bottomLab.setBorder(BorderFactory.createEmptyBorder(10, 5, 10, 5));
    bottomLab.setLayout(gb);

    constraints.gridx = 0;
    constraints.gridy = 0;
    constraints.weightx = 5;
    constraints.fill = GridBagConstraints.HORIZONTAL;
    constraints.gridwidth = 1;
    constraints.gridheight = 1;
    constraints.insets = new Insets(0, 2, 0, 2);
    bottomLab.add(m_UpBut, constraints);
    constraints.gridx = 1;
    constraints.gridy = 0;
    constraints.weightx = 5;
    constraints.gridwidth = 1;
    constraints.gridheight = 1;
    bottomLab.add(m_DownBut, constraints);

    add(topLab, BorderLayout.NORTH);
    add(new JScrollPane(m_List), BorderLayout.CENTER);
    add(bottomLab, BorderLayout.SOUTH);
  }

  /**
   * sets the state of the buttons according to the selection state of the
   * JList.
   * 
   * @param e the event
   */
  private void setButtons(ListSelectionEvent e) {
    if ((e == null) || (e.getSource() == m_List)) {
      m_DeleteBut.setEnabled(m_List.getSelectedIndex() > -1);
      m_EditBut.setEnabled(m_List.getSelectedIndices().length == 1);
      m_UpBut.setEnabled(JListHelper.canMoveUp(m_List));
      m_DownBut.setEnabled(JListHelper.canMoveDown(m_List));
    }
  }

  /**
   * Tells the panel to act on a new experiment.
   * 
   * @param exp a value of type 'Experiment'
   */
  public void setExperiment(Experiment exp) {

    m_Exp = exp;
    m_List.setModel(m_Exp.getDatasets());
    m_AddBut.setEnabled(true);
    setButtons(null);
  }

  /**
   * Gets all the files in the given directory that match the currently selected
   * extension.
   * 
   * @param directory the directory to get the files for
   * @param files the list to add the files to
   */
  protected void getFilesRecursively(File directory, Vector<File> files) {

    try {
      String[] currentDirFiles = directory.list();
      for (int i = 0; i < currentDirFiles.length; i++) {
        currentDirFiles[i] = directory.getCanonicalPath() + File.separator
          + currentDirFiles[i];
        File current = new File(currentDirFiles[i]);
        if (m_FileChooser.getFileFilter().accept(current)) {
          if (current.isDirectory()) {
            getFilesRecursively(current, files);
          } else {
            files.addElement(current);
          }
        }
      }
    } catch (Exception e) {
      System.err.println("IOError occured when reading list of files");
    }
  }

  
  /**
   * Gets all the files in the given directory that match the currently selected
   * extension.
   * 
   * @param directory the directory to get the files for
   * @param files the list to add the files to
   */
  protected void getTraTstKEELSamplesFilesContext(File directory, Vector<File> files, String st_context, String[] currentDirFiles) {

	  try {
		  File currentSubdir;

		  if (currentDirFiles.length != 1)
			  System.err.println("Not found '" + st_context + "' data sets folder!");
		  currentSubdir = new File(directory, currentDirFiles[0]);
		  File[] contextDatasetsFolders = currentSubdir.listFiles();
		  int counter = 0;
		  for (int i = 0; i < contextDatasetsFolders.length; i++)
			  if (contextDatasetsFolders[i].isDirectory()) {
				  File[] currentDatasetTraSamples;
				  currentDatasetTraSamples = contextDatasetsFolders[i].listFiles((dir, name) -> name.endsWith("-5-1tra.dat"));
				  // The filename's pattern is different for '3.imbalanced-preprocessed'context
				  if ((currentDatasetTraSamples == null) || (currentDatasetTraSamples.length == 0))
					  currentDatasetTraSamples = contextDatasetsFolders[i].listFiles((dir, name) -> name.endsWith("0s0.tra.dat"));
				  if ((currentDatasetTraSamples == null) || (currentDatasetTraSamples.length != 1))
					  System.err.println("Not found the first training sample of '" + currentDatasetTraSamples[i].getName() + "' data set in '" + st_context + "' data sets context!");
				  else {
					  FileFilter[] fileFilters = m_FileChooser.getChoosableFileFilters();
					  for (FileFilter filter: fileFilters)
						  if (filter.accept(currentDatasetTraSamples[0])) {
							  files.addElement(currentDatasetTraSamples[0]);
							  //System.out.println(contextDatasetsFolders[i].getName() + ": " + currentDatasetTraSamples[0].getName());
							  counter++;
							  break;
						  }
				  }
			  }
		  System.out.println("'" + currentDirFiles[0] + "': " + counter + " data sets found!");
	  } catch (Exception e) {
		  System.err.println("IOError occured when reading list of files");
	  }
  }

  /**
   * Gets all the files in the given directory that match the currently selected
   * extension.
   * 
   * @param directory the directory to get the files for
   * @param files the list to add the files to
   */
  protected void getTraTstKEELSamplesFilesRecursively(File directory, Vector<File> files) {

		  String[] currentDirFiles;
		  String[] currentDir = new String[]{"."};
		  // Checks if exists '1.standard' folder
		  currentDirFiles = directory.list((dir, name) -> name.contains("standard"));
		  if (currentDirFiles.length > 0)
			  getTraTstKEELSamplesFilesContext(directory, files, "standard", currentDirFiles);
		  // or if '1.standard' is the parent directory
		  currentDirFiles = directory.getParentFile().list((dir, name) -> name.contains("standard"));
		  if (currentDirFiles.length > 0) {
			  getTraTstKEELSamplesFilesContext(directory, files, "standard", currentDir);
			  return;
		  }
		  
		  // Checks if exists '2.imbalanced' folder
		  currentDirFiles = directory.list((dir, name) -> name.contains("imbalanced") && 
				  !(name.contains("preprocessed") || name.contains("SMOTE")));
		  if (currentDirFiles.length > 0)
			  getTraTstKEELSamplesFilesContext(directory, files, "imbalanced", currentDirFiles);
		  // or if '2.imbalanced' is the parent directory
		  currentDirFiles = directory.getParentFile().list((dir, name) -> name.contains("imbalanced") && 
				  !(name.contains("preprocessed") || name.contains("SMOTE")));
		  if (currentDirFiles.length > 0) {
			  getTraTstKEELSamplesFilesContext(directory.getParentFile(), files, "imbalanced", currentDir);
			  return;
		  }

		  // Checks if exists '3.imbalanced-preprocessed' folder
		  currentDirFiles = directory.list((dir, name) -> name.contains("imbalanced") && 
				  (name.contains("preprocessed") || name.contains("SMOTE")));
		  if (currentDirFiles.length > 0)
			  getTraTstKEELSamplesFilesContext(directory, files, "imbalanced-preprocessed", currentDirFiles);
		  // or if '3.imbalanced-preprocessed' is the parent directory
		  currentDirFiles = directory.getParentFile().list((dir, name) -> name.contains("imbalanced") && 
				  (name.contains("preprocessed") || name.contains("SMOTE")));
		  if (currentDirFiles.length > 0) {
			  getTraTstKEELSamplesFilesContext(directory.getParentFile(), files, "imbalanced-preprocessed", currentDir);
			  return;
		  }
  }

  /**
   * Handle actions when buttons get pressed.
   * 
   * @param e a value of type 'ActionEvent'
   */
  @Override
  public void actionPerformed(ActionEvent e) {
    boolean useRelativePaths = m_relativeCheck.isSelected();

    if (e.getSource() == m_AddBut) {
      // Let the user select an arff file from a file chooser
      int returnVal = m_FileChooser.showOpenDialog(this);
      if (returnVal == JFileChooser.APPROVE_OPTION) {
        if (m_FileChooser.isMultiSelectionEnabled()) {
          File[] selected = m_FileChooser.getSelectedFiles();
          for (File element : selected) {
            if (element.isDirectory()) {
              Vector<File> files = new Vector<File>();
              getFilesRecursively(element, files);

              // sort the result
              Collections.sort(files);

              for (int j = 0; j < files.size(); j++) {
                File temp = files.elementAt(j);
                if (useRelativePaths) {
                  try {
                    temp = Utils.convertToRelativePath(temp);
                  } catch (Exception ex) {
                    ex.printStackTrace();
                  }
                }
                m_Exp.getDatasets().addElement(temp);
              }
            } else {
              File temp = element;
              if (useRelativePaths) {
                try {
                  temp = Utils.convertToRelativePath(temp);
                } catch (Exception ex) {
                  ex.printStackTrace();
                }
              }
              m_Exp.getDatasets().addElement(temp);
            }
          }
          setButtons(null);
        } else {
          if (m_FileChooser.getSelectedFile().isDirectory()) {
            Vector<File> files = new Vector<File>();
            if (m_TraTstKEELSamples) {
            	getTraTstKEELSamplesFilesRecursively(m_FileChooser.getSelectedFile(), files);
            	for (File f: files)
            		m_Exp.getDatasets().addElement(f);
            } else {
                getFilesRecursively(m_FileChooser.getSelectedFile(), files);

                // sort the result
                Collections.sort(files);

                for (int j = 0; j < files.size(); j++) {
                  File temp = files.elementAt(j);
                  if (useRelativePaths) {
                    try {
                      temp = Utils.convertToRelativePath(temp);
                    } catch (Exception ex) {
                      ex.printStackTrace();
                    }
                  }
                  m_Exp.getDatasets().addElement(temp);
                }
            }
          } else {
        	  if (!m_TraTstKEELSamples) {
                  File temp = m_FileChooser.getSelectedFile();
                  if (useRelativePaths) {
                    try {
                      temp = Utils.convertToRelativePath(temp);
                    } catch (Exception ex) {
                      ex.printStackTrace();
                    }
                  }
                  m_Exp.getDatasets().addElement(temp);
        	  }
          }
          setButtons(null);
        }
      }
    } else if (e.getSource() == m_DeleteBut) {
      // Delete the selected files
      int[] selected = m_List.getSelectedIndices();
      if (selected != null) {
        for (int i = selected.length - 1; i >= 0; i--) {
          int current = selected[i];
          m_Exp.getDatasets().removeElementAt(current);
          if (m_Exp.getDatasets().size() > current) {
            m_List.setSelectedIndex(current);
          } else {
            m_List.setSelectedIndex(current - 1);
          }
        }
      }
      setButtons(null);
    } else if (e.getSource() == m_EditBut) {
      // Delete the selected files
      int selected = m_List.getSelectedIndex();
      if (selected != -1) {
        ViewerDialog dialog = new ViewerDialog(null);
        String filename = m_List.getSelectedValue().toString();
        int result;
        try {
          DataSource source = new DataSource(filename);
          result = dialog.showDialog(source.getDataSet());
          // nasty workaround for Windows regarding locked files:
          // if file Reader in Loader is not closed explicitly, we cannot
          // overwrite the file.
          source = null;
          System.gc();
          // workaround end
          if ((result == ViewerDialog.APPROVE_OPTION) && (dialog.isChanged())) {
            result = JOptionPane.showConfirmDialog(this,
              "File was modified - save changes?");
            if (result == JOptionPane.YES_OPTION) {
              Saver saver = ConverterUtils.getSaverForFile(filename);
              saver.setFile(new File(filename));
              saver.setInstances(dialog.getInstances());
              saver.writeBatch();
            }
          }
        } catch (Exception ex) {
          JOptionPane.showMessageDialog(this, "Error loading file '" + filename
            + "':\n" + ex.toString(), "Error loading file",
            JOptionPane.INFORMATION_MESSAGE);
        }
      }
      setButtons(null);
    } else if (e.getSource() == m_UpBut) {
      JListHelper.moveUp(m_List);
    } else if (e.getSource() == m_DownBut) {
      JListHelper.moveDown(m_List);
    }
  }

  /**
   * Tests out the dataset list panel from the command line.
   * 
   * @param args ignored
   */
  public static void main(String[] args) {

    try {
      final JFrame jf = new JFrame("Dataset List Editor");
      jf.getContentPane().setLayout(new BorderLayout());
      DatasetListPanel dp = new DatasetListPanel();
      jf.getContentPane().add(dp, BorderLayout.CENTER);
      jf.addWindowListener(new WindowAdapter() {
        @Override
        public void windowClosing(WindowEvent e) {
          jf.dispose();
          System.exit(0);
        }
      });
      jf.pack();
      jf.setVisible(true);
      System.err.println("Short nap");
      Thread.sleep(3000);
      System.err.println("Done");
      dp.setExperiment(new Experiment());
    } catch (Exception ex) {
      ex.printStackTrace();
      System.err.println(ex.getMessage());
    }
  }

  /**
   * Sets if the user has selected a experiment with training/test KEEL samples or not
   * 
   * @param b indicates if the experiment uses training/test KEEL samples or not
   */
  protected void setTraTstKEELSamples(boolean b) {
	  m_TraTstKEELSamples = b;
  }

}
