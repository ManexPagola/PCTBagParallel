/**
 * 
 */
package weka.experiment;

import java.io.File;

import weka.core.Instances;
import weka.core.converters.AbstractFileSaver;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;

/**
 * "Merge tra/tst KEEL samples to ARFF" based on CrossValidationKEELFromTraTstResultProducer, 
 * it uses the training and test samples already generated (using KEEL format) from the first fold of 
 * a 5-fold cross-validation to reconstruct the complete sample and save it in ARFF format.
 * The ZeroR classifier can be used, since no results are intended to be obtained.
 * 
 */
public class CrossValidationKEELFromTraTstToARFFResultProducer extends CrossValidationKEELFromTraTstResultProducer {

	private static final long serialVersionUID = 2772933355948124375L;

	@Override
	public void doRun(int run) throws Exception {

		Instances inst = new Instances(m_Instances);

		// Code based on copied from weka.classifiers.bayes.net.GUI.ActionGenerateData.actionPerformed(ActionEvent)
		String dir = m_fDataset.getParentFile().getName();
		String sname;
		if (m_fDataset.getParentFile().getParentFile().getName().contentEquals("."))
			sname = m_fDataset.getParentFile().getParentFile().getParentFile().getAbsolutePath()+ File.separator + dir +".dat.arff";
		else
			sname = m_fDataset.getParentFile().getParentFile().getAbsolutePath()+ File.separator + dir +".dat.arff";
		AbstractFileSaver saver = ConverterUtils.getSaverForFile(sname);
		// no idea what the format is, so let's save it as ARFF file
		if (saver == null) {
			System.out.println("saver == null!");
			saver = new ArffSaver();
		}
		saver.setFile(new File(sname));
		try {
			saver.setInstances(inst);
			saver.writeBatch();
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}
}
