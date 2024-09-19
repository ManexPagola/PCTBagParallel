package weka.classifiers.trees.j48Consolidated;

import java.util.ArrayList;

import weka.classifiers.trees.j48.C45PruneableClassifierTree;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class for handling a consolidated tree structure that can
 * be pruned using C4.5 procedures.
 * *************************************************************************************
 * 
 * @author Jes&uacute;s M. P&eacute;rez (txus.perez@ehu.eus) 
 * @version $Revision: 1.2 $
 */
public class C45ConsolidatedPruneableClassifierTree extends
		C45PruneableClassifierTree {

	/** for serialization */
	private static final long serialVersionUID = 2660972525647728377L;

	/**
	 * Constructor for pruneable consolidated tree structure. Calls
	 * the superclass constructor.
	 *
	 * @param toSelectLocModel selection method for local splitting model
	 * @param pruneTree true if the tree is to be pruned
	 * @param cf the confidence factor for pruning
	 * @param raiseTree true if subtree raising has to be performed
	 * @param cleanup true if cleanup has to be done
	 * @param collapseTree true if collapse has to be done
	 * @throws Exception if something goes wrong
	 */
	public C45ConsolidatedPruneableClassifierTree(
			ModelSelection toSelectLocModel, boolean pruneTree, float cf,
			boolean raiseTree, boolean cleanup, boolean collapseTree) throws Exception {
		super(toSelectLocModel, pruneTree, cf, raiseTree, cleanup, collapseTree);
	}

	/**
	 * Method for building a pruneable classifier consolidated tree.
	 *
	 * @param data the data for pruning the consolidated tree
	 * @param samplesVector the vector of samples for building the consolidated tree
	 * @throws Exception if something goes wrong
	 */
	public void buildClassifier(Instances data, Instances[] samplesVector) throws Exception {

		buildTree(data, samplesVector, m_subtreeRaising || !m_cleanup);
		if (m_collapseTheTree) {
			collapse();
		}
		if (m_pruneTheTree) {
			prune();
		}
		if (m_cleanup) {
			cleanup(new Instances(data, 0));
		}
	}
	
	/**
	 * Returns a newly created tree.
	 *
	 * @param data the data to work with
	 * @param samplesVector the vector of samples for building the consolidated tree
	 * @return the new consolidated tree
	 * @throws Exception if something goes wrong
	 */
	protected ClassifierTree getNewTree(Instances data, Instances[] samplesVector) throws Exception {

		C45ConsolidatedPruneableClassifierTree newTree = 
				new C45ConsolidatedPruneableClassifierTree(m_toSelectModel, m_pruneTheTree, m_CF,
						m_subtreeRaising, m_cleanup, m_collapseTheTree);
		newTree.buildTree(data, samplesVector, m_subtreeRaising);

		return newTree;
	}

	/**
	 * Builds the consolidated tree structure.
	 * (based on the method buildTree() of the class 'ClassifierTree')
	 *
	 * @param data the data for pruning the consolidated tree
	 * @param samplesVector the vector of samples used for consolidation
	 * @param keepData is training data to be kept?
	 * @throws Exception if something goes wrong
	 */
	public void buildTree(Instances data, Instances[] samplesVector, boolean keepData) throws Exception {
		/** Number of Samples. */
		int numberSamples = samplesVector.length;
	

		if (keepData) {
			m_train = data;
		}
		m_test = null;
		m_isLeaf = false;
		m_isEmpty = false;
		m_sons = null;

		m_localModel = ((C45ConsolidatedModelSelection)m_toSelectModel).selectModel(data, samplesVector);

		if (m_localModel.numSubsets() > 1) {
			/** Vector storing the obtained subsamples after the split of data */
			Instances [] localInstances;
			/** Vector storing the obtained subsamples after the split of each sample of the vector */
			ArrayList<Instances []> localInstancesVector = new ArrayList<Instances []>();

			localInstances = m_localModel.split(data);
			for (int iSamples = 0; iSamples < numberSamples; iSamples++)
				localInstancesVector.add(m_localModel.split(samplesVector[iSamples]));
			data = null;
			samplesVector = null;
			m_sons = new ClassifierTree [m_localModel.numSubsets()];
			for (int iSon = 0; iSon < m_sons.length; iSon++) {
				/** Vector storing the subsamples related to the iSon-th son */
				Instances[] localSamplesVector = new Instances[numberSamples];
				for (int iSamples = 0; iSamples < numberSamples; iSamples++)
					localSamplesVector[iSamples] = 
							((Instances[]) localInstancesVector.get(iSamples))[iSon];
				m_sons[iSon] = getNewTree(localInstances[iSon], localSamplesVector);
				localInstances[iSon] = null;
				localSamplesVector = null;
			}
			localInstances = null;
			localInstancesVector.clear();
			localInstancesVector = null;
		}else{
			m_isLeaf = true;
			if (Utils.eq(m_localModel.distribution().total(), 0))
				m_isEmpty = true;
			data = null;
			samplesVector = null;
		}
	}

}
