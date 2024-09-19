/**
 * 
 */
package weka.experiment;

import java.io.File;
import java.util.Random;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.WekaException;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * "Cross-validation by merging tra/tst KEEL samples" Based on CrossValidation1x5KEELResultProducer, 
 * it uses the training and test samples already generated (using KEEL format) from the first fold of
 * a 5-fold cross-validation to rebuild the entire sample and allow the user to run multiple 
 * 5-fold cross-validation runs.
 * 
 */
public class CrossValidationKEELFromTraTstResultProducer extends CrossValidation1x5KEELResultProducer {

	private static final long serialVersionUID = 252138613356064807L;

	/**
	 * Complete the data (training sample) with the test sample
	 * @throws Exception 
	 */
	protected void completeDatawithTestSample() throws Exception {
		// test set
		String filename;
		File file;
		Instances test;
		int fold = 0;
		// The filename's pattern is different for '3.imbalanced-preprocessed' context
		filename = (m_iContext == 3) ? m_pathDB + m_basenameDB + fold + "s0.tst.dat" : m_pathDB + m_basenameDB + (fold + 1) + "tst.dat";
		file =  new File(filename);
		if (!file.exists()) {
			throw new WekaException("Test set '" + filename + "' not found!");
		}
		test = DataSource.read(filename);
		// test headers
		if (!m_Instances.equalHeaders(test)) {
			throw new WekaException("Train and test set (= " + filename + ") "
					+ "are not compatible:\n" + m_Instances.equalHeadersMsg(test));
		}
		m_Instances.addAll(test);
		String DBname = m_iContext + "-" + m_dirDB;
		m_Instances.setRelationName(DBname);
	}

	@Override
	public void doRun(int run) throws Exception {

		if (getRawOutput()) {
			if (m_ZipDest == null) {
				m_ZipDest = new OutputZipper(m_OutputFile);
			}
		}

		if (m_Instances == null) {
			throw new Exception("No Instances set");
		}
		// Randomize on a copy of the original dataset
		Instances runInstances = new Instances(m_Instances);
		Random random = new Random(run);
		runInstances.randomize(random);
		if (runInstances.classAttribute().isNominal()) {
			runInstances.stratify(m_NumFolds);
		}
		for (int fold = 0; fold < m_NumFolds; fold++) {
			// Add in some fields to the key like run and fold number, dataset name
			Object[] seKey = m_SplitEvaluator.getKey();
			Object[] key = new Object[seKey.length + 3];
			//key[0] = Utils.backQuoteChars(m_Instances.relationName());
			// In '3.imbalanced-preprocessed' context, for all databases the names of the samples are the same.
			//key[0] = (i_context == 3) ? i_context + "-" + datasetDir : i_context + "-" + m_fDataset.getName();
			key[0] = m_iContext + "-" + m_dirDB;
			key[1] = "" + run;
			key[2] = "" + (fold + 1);
			System.arraycopy(seKey, 0, key, 3, seKey.length);
			if (m_ResultListener.isResultRequired(this, key)) {
				Instances train = runInstances.trainCV(m_NumFolds, fold, random);
				Instances test = runInstances.testCV(m_NumFolds, fold);
				try {
					Object[] seResults = m_SplitEvaluator.getResult(train, test);
					Object[] results = new Object[seResults.length + 1];
					results[0] = getTimestamp();
					System.arraycopy(seResults, 0, results, 1, seResults.length);
					if (m_debugOutput) {
						String resultName = ("" + run + "." + (fold + 1) + "."
								+ Utils.backQuoteChars(runInstances.relationName()) + "." + m_SplitEvaluator
								.toString()).replace(' ', '_');
						resultName = Utils.removeSubstring(resultName, "weka.classifiers.");
						resultName = Utils.removeSubstring(resultName, "weka.filters.");
						resultName = Utils.removeSubstring(resultName,
								"weka.attributeSelection.");
						m_ZipDest.zipit(m_SplitEvaluator.getRawResultOutput(), resultName);
					}
					m_ResultListener.acceptResult(this, key, results);
				} catch (Exception ex) {
					// Save the train and test datasets for debugging purposes?
					throw ex;
				}
			}
		}
	}
}
