/**
 *
 */
package weka.classifiers.trees.j48ItPartiallyConsolidated;

import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.j48PartiallyConsolidated.C45ModelSelectionExtended;
import weka.classifiers.trees.j48PartiallyConsolidated.C45PartiallyConsolidatedPruneableClassifierTree;
import weka.classifiers.trees.j48PartiallyConsolidated.C45PruneableClassifierTreeExtended;

import java.util.ArrayList;

import weka.classifiers.trees.J48It;
import weka.classifiers.trees.J48ItPartiallyConsolidated;
import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class for handling a consolidated tree structure that can be pruned using
 * C4.5 procedures.
 * *************************************************************************************
 * Attention! Removed 'final' modifier from collapse() function of
 * j48/C45PruneableClassifierTree class and from cleanup() function of
 * j48/ClassifierTree class in order to overwrite these functions here.
 * *************************************************************************************<br/>
 *
 * @author Josué Cabezas Regoyo
 * @author Jesús M. Pérez (txus.perez@ehu.eus)
 * @version $Revision: 0.3 $
 */
public class C45ItPartiallyConsolidatedPruneableClassifierTree extends C45PartiallyConsolidatedPruneableClassifierTree {

	/** for serialization **/
	private static final long serialVersionUID = 6410655550027990502L;

	/** Indicates the order in which the node was treated */
	private int m_order;

	/**
	 * Builds the tree up to a maximum of depth levels. Set m_maximumLevel to 0 for
	 * default.
	 */
	private int m_maximumCriteria;

	/** Indicates the criteria that should be used to build the tree */
	private int m_priorityCriteria;

	/** True if the partial Consolidated tree(CT) is to be pruned. */
	protected boolean m_pruneTheConsolidatedTree = false;
	
	/** True if the partial Consolidated tree(CT) is to be collapsed. */
	protected boolean m_collapseTheCTree = false;

	/** Time taken to build the whole Consolidated Tree (CT), including pruning/collapsing, if required. */
	protected double m_elapsedTimeTrainingWholeCT = (double)Double.NaN;

	/** Time taken to build the partial Consolidated Tree (CT), including pruning/collapsing, if required. */
	protected double m_elapsedTimeTrainingPartialCT = (double)Double.NaN;

	/** Time taken to build all base trees that compose the final multiple classifier (Bagging), including pruning/collapsing, if required. */
	protected double m_elapsedTimeTrainingAssocBagging = (double)Double.NaN;

	/**
	 * Constructor for pruneable consolidated tree structure. Calls the superclass
	 * constructor.
	 *
	 * @param toSelectLocModel      selection method for local splitting model
	 * @param pruneTree             true if the tree is to be pruned
	 * @param cf                    the confidence factor for pruning
	 * @param raiseTree             true if subtree raising has to be performed
	 * @param cleanup               true if cleanup has to be done
	 * @param collapseTree          true if collapse has to be done
	 * @param numberSamples         Number of Samples
	 * @param ITPCTmaximumCriteria  maximum number of nodes or levels
	 * @param ITPCTpriorityCriteria criteria to build the tree
	 * @param pruneCT true if the CT tree is to be pruned
	 * @param collapseCT true if the CT tree is to be collapsed
	 * @throws Exception if something goes wrong
	 */
	public C45ItPartiallyConsolidatedPruneableClassifierTree(
			ModelSelection toSelectLocModel, C45ModelSelectionExtended baseModelToForceDecision,
			boolean pruneTree, float cf,
			boolean raiseTree, boolean cleanup, 
			boolean collapseTree, int numberSamples,
			boolean notPreservingStructure,
			int ITPCTpriorityCriteria, boolean pruneCT, boolean collapseCT) throws Exception {
		super(toSelectLocModel, baseModelToForceDecision, pruneTree, cf, raiseTree, cleanup, collapseTree,
				numberSamples, notPreservingStructure);

		// Initialize each criteria
		m_priorityCriteria = ITPCTpriorityCriteria;
		m_pruneTheConsolidatedTree = pruneCT;
		m_collapseTheCTree = collapseCT;
	}

	/**
	 * Method for building a pruneable classifier consolidated tree.
	 *
	 * @param data                 the data for pruning the consolidated tree
	 * @param samplesVector        the vector of samples for building the
	 *                             consolidated tree
	 * @param consolidationPercent the value of consolidation percent
	 * @throws Exception if something goes wrong
	 */
	public void buildClassifier(Instances data, Instances[] samplesVector, float consolidationPercent,
			int consolidationNumberHowToSet) throws Exception {
		long trainTimeStart = 0, trainTimeElapsed = 0;
		if (m_priorityCriteria == J48It.Original) {

			m_pruneTheTree = m_pruneTheConsolidatedTree;
			m_collapseTheTree = m_collapseTheCTree;
			super.buildClassifier(data, samplesVector, consolidationPercent);

		} else {
			if (consolidationNumberHowToSet == J48ItPartiallyConsolidated.ConsolidationNumber_Percentage) {
								
				trainTimeStart = System.currentTimeMillis();
				super.buildTree(data, samplesVector, m_subtreeRaising || !m_cleanup); // build the tree without restrictions
								
				if (m_collapseTheCTree) {
					collapse();
				}
				if (m_pruneTheConsolidatedTree) {
					prune();
				}
				trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
				System.out.println("Time taken to build the whole consolidated tree: " + Utils.doubleToString(trainTimeElapsed / 1000.0, 2) + " seconds\n");
				m_elapsedTimeTrainingWholeCT = trainTimeElapsed / (double)1000.0;

				if (m_priorityCriteria == J48It.Levelbylevel) {

					// Number of levels of the consolidated tree
					int treeLevels = numLevels();

					// Number of levels of the consolidated tree to leave as consolidated based on
					// given consolidationPercent
					int numberLevelsConso = (int) (((treeLevels * consolidationPercent) / 100) + 0.5);
					m_maximumCriteria = numberLevelsConso;
					setNumInternalNodesConso(numberLevelsConso);
					System.out.println(
							"Number of levels to leave as consolidated: " + numberLevelsConso + " of " + treeLevels);

				} else {

					// Number of internal nodes of the consolidated tree
					int innerNodes = numNodes() - numLeaves();

					// Number of nodes of the consolidated tree to leave as consolidated based on
					// given consolidationPercent
					int numberNodesConso = (int) (((innerNodes * consolidationPercent) / 100) + 0.5);
					m_maximumCriteria = numberNodesConso;
					setNumInternalNodesConso(numberNodesConso);
					System.out.println(
							"Number of nodes to leave as consolidated: " + numberNodesConso + " of " + innerNodes);

				}

			} else // consolidationNumberHowToSet ==
					// J48ItPartiallyConsolidated.ConsolidationNumber_Value
			{
				m_maximumCriteria = (int) consolidationPercent;
				System.out.println("Number of nodes or levels to leave as consolidated: " + m_maximumCriteria);
				m_elapsedTimeTrainingWholeCT = (double)0.0;
			}

			// buildTree
			trainTimeStart = System.currentTimeMillis();
			buildTree(data, samplesVector, m_subtreeRaising || !m_cleanup);
			if (m_collapseTheCTree) {
				collapse();
			}
			if (m_pruneTheConsolidatedTree) {
				prune();
			}
			trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
			System.out.println("Time taken to build the partial consolidated tree: " + Utils.doubleToString(trainTimeElapsed / 1000.0, 2) + " seconds\n");
			m_elapsedTimeTrainingPartialCT = trainTimeElapsed / (double)1000.0;

			trainTimeStart = System.currentTimeMillis();
			applyBagging();
			trainTimeElapsed = System.currentTimeMillis() - trainTimeStart;
			System.out.println("Time taken to build the associated Bagging: " + Utils.doubleToString(trainTimeElapsed / 1000.0, 2) + " seconds\n");
			m_elapsedTimeTrainingAssocBagging = trainTimeElapsed / (double)1000.0;

			if (m_cleanup)
				cleanup(new Instances(data, 0));
		}
	}

	/**
	 * Builds the consolidated tree structure. (based on the method buildTree() of
	 * the class 'ClassifierTree')
	 *
	 * @param data          the data for pruning the consolidated tree
	 * @param samplesVector the vector of samples used for consolidation
	 * @param keepData      is training data to be kept?
	 * @throws Exception if something goes wrong
	 */
	public void buildTree(Instances data, Instances[] samplesVector, boolean keepData) throws Exception {
		/** Number of Samples. */
		int numberSamples = samplesVector.length;

		/** Initialize the consolidated tree */
		if (keepData) {
			m_train = data;
		}
		m_test = null;
		m_isLeaf = false;
		m_isEmpty = false;
		m_sons = null;
		/** Initialize the base trees */
		for (int iSample = 0; iSample < numberSamples; iSample++)
			m_sampleTreeVector[iSample].initiliazeTree(samplesVector[iSample], keepData);

		ArrayList<Object[]> list = new ArrayList<>();

		// add(Data, samplesVector, tree, orderValue, currentLevel)
		list.add(new Object[] { data, samplesVector, this, null, 0 }); // The parent node is considered level 0

		int index = 0;
		double orderValue;

		int internalNodes = 0;

		while (list.size() > 0) {

			Object[] current = list.get(0);

			/** Current node level. **/
			int currentLevel = (int) current[4];

			/** Number of Samples. */
			Instances[] currentSamplesVector = (Instances[]) current[1];
			//int numberSamples = currentSamplesVector.length;

			list.set(0, null); // Null to free up memory
			list.remove(0);

			Instances currentData = (Instances) current[0];
			C45ItPartiallyConsolidatedPruneableClassifierTree currentTree = (C45ItPartiallyConsolidatedPruneableClassifierTree) current[2];
			currentTree.m_order = index;

			/** Initialize the consolidated tree */
			if (keepData) {
				currentTree.m_train = currentData;
			}
			currentTree.m_test = null;
			currentTree.m_isLeaf = false;
			currentTree.m_isEmpty = false;
			currentTree.m_sons = null;

			/** Initialize the base trees */
			for (int iSample = 0; iSample < numberSamples; iSample++)
				currentTree.m_sampleTreeVector[iSample].initiliazeTree(currentSamplesVector[iSample], keepData);

			/**
			 * Select the best model to split (if it is worth) based on the consolidation
			 * proccess
			 */
			currentTree.m_localModel = ((C45ConsolidatedModelSelection) currentTree.m_toSelectModel)
					.selectModel(currentData, currentSamplesVector);
			for (int iSample = 0; iSample < numberSamples; iSample++)
				currentTree.m_sampleTreeVector[iSample].setLocalModel(currentSamplesVector[iSample],
						currentTree.m_localModel);

			if ((currentTree.m_localModel.numSubsets() > 1) && ((m_priorityCriteria == J48It.Original)
					|| ((m_priorityCriteria == J48It.Levelbylevel) && (currentLevel < m_maximumCriteria))
					|| ((m_priorityCriteria > J48It.Levelbylevel) && (internalNodes < m_maximumCriteria)))) {

				/** Vector storing the obtained subsamples after the split of data */
				Instances[] localInstances;
				/**
				 * Vector storing the obtained subsamples after the split of each sample of the
				 * vector
				 */
				ArrayList<Instances[]> localInstancesVector = new ArrayList<Instances[]>();

				/**
				 * For some base trees, although the current node is not a leaf, it could be
				 * empty. This is necessary in order to calculate correctly the class membership
				 * probabilities for the given test instance in each base tree
				 */

				ArrayList<Object[]> listSons = new ArrayList<>();

				for (int iSample = 0; iSample < numberSamples; iSample++)
					if (Utils.eq(currentTree.m_sampleTreeVector[iSample].getLocalModel().distribution().total(), 0))
						currentTree.m_sampleTreeVector[iSample].setIsEmpty(true);

				/** Split data according to the consolidated m_localModel */
				localInstances = currentTree.m_localModel.split(currentData);
				for (int iSample = 0; iSample < numberSamples; iSample++)
					localInstancesVector.add(currentTree.m_localModel.split(currentSamplesVector[iSample]));

				/**
				 * Create the child nodes of the current node and call recursively to
				 * getNewTree()
				 */
				currentData = null;
				currentSamplesVector = null;
				currentTree.m_sons = new ClassifierTree[currentTree.m_localModel.numSubsets()];
				for (int iSample = 0; iSample < numberSamples; iSample++)
					((C45PruneableClassifierTreeExtended) currentTree.m_sampleTreeVector[iSample])
							.createSonsVector(currentTree.m_localModel.numSubsets());

				//////////////////
				C45ModelSelectionExtended baseModelToForceDecision = 
						currentTree.m_sampleTreeVector[0].getBaseModelToForceDecision();
				for (int iSon = 0; iSon < currentTree.m_sons.length; iSon++) {
					/** Vector storing the subsamples related to the iSon-th son */
					Instances[] localSamplesVector = new Instances[numberSamples];
					for (int iSample = 0; iSample < numberSamples; iSample++)
						localSamplesVector[iSample] = ((Instances[]) localInstancesVector.get(iSample))[iSon];

					// getNewTree
					C45ItPartiallyConsolidatedPruneableClassifierTree newTree = new C45ItPartiallyConsolidatedPruneableClassifierTree(
							currentTree.m_toSelectModel, baseModelToForceDecision, m_pruneTheTree, m_CF,
							m_subtreeRaising, m_cleanup, m_collapseTheTree, localSamplesVector.length,
							m_pruneBaseTreesWithoutPreservingConsolidatedStructure,
							m_priorityCriteria, m_pruneTheConsolidatedTree, m_collapseTheCTree);

					/** Set the recent created base trees like the sons of the given parent node */
					for (int iSample = 0; iSample < numberSamples; iSample++)
						((C45PruneableClassifierTreeExtended) currentTree.m_sampleTreeVector[iSample]).setIthSon(iSon,
								newTree.m_sampleTreeVector[iSample]);

					if (m_priorityCriteria == J48ItPartiallyConsolidated.Size) // Added by size, largest to smallest
					{

						orderValue = currentTree.m_localModel.distribution().perBag(iSon);

						Object[] son = new Object[] { localInstances[iSon], localSamplesVector, newTree, orderValue,
								currentLevel + 1 };
						addSonOrderedByValue(list, son);

					} else if (m_priorityCriteria == J48ItPartiallyConsolidated.Gainratio) // Added by gainratio,
																							// largest to smallest
					{
						ClassifierSplitModel sonModel = ((C45ItPartiallyConsolidatedPruneableClassifierTree) newTree).m_toSelectModel
								.selectModel(localInstances[iSon]);
						if (sonModel.numSubsets() > 1) {

							orderValue = ((C45Split) sonModel).gainRatio();

						} else {

							orderValue = (double) Double.MIN_VALUE;
						}
						Object[] son = new Object[] { localInstances[iSon], localSamplesVector, newTree, orderValue,
								currentLevel + 1 };
						addSonOrderedByValue(list, son);

					} else if (m_priorityCriteria == J48ItPartiallyConsolidated.Gainratio_normalized) // Added by
																										// gainratio
																										// normalized,
					// largest to smallest
					{

						double size = currentTree.m_localModel.distribution().perBag(iSon);
						double gainRatio;
						ClassifierSplitModel sonModel = ((C45ItPartiallyConsolidatedPruneableClassifierTree) newTree).m_toSelectModel
								.selectModel(localInstances[iSon]);
						if (sonModel.numSubsets() > 1) {

							gainRatio = ((C45Split) sonModel).gainRatio();
							orderValue = size * gainRatio;

						} else {

							orderValue = (double) Double.MIN_VALUE;
						}
						Object[] son = new Object[] { localInstances[iSon], localSamplesVector, newTree, orderValue,
								currentLevel + 1 };
						addSonOrderedByValue(list, son);

					} else {
						listSons.add(new Object[] { localInstances[iSon], localSamplesVector, newTree, 0,
								currentLevel + 1 });
					}

					currentTree.m_sons[iSon] = newTree;

					localInstances[iSon] = null;
					localSamplesVector = null;
				}

				if (m_priorityCriteria == J48ItPartiallyConsolidated.Levelbylevel) { // Level by level
					list.addAll(listSons);
				}

				else if (m_priorityCriteria == J48ItPartiallyConsolidated.Preorder
						|| m_priorityCriteria == J48ItPartiallyConsolidated.Original) { // Preorder
					listSons.addAll(list);
					list = listSons;
				}

				localInstances = null;
				localInstancesVector.clear();
				listSons = null;
				internalNodes++;

			} else {
				currentTree.m_isLeaf = true;
				for (int iSample = 0; iSample < numberSamples; iSample++)
					currentTree.m_sampleTreeVector[iSample].setIsLeaf(true);

				if (Utils.eq(currentTree.m_localModel.distribution().total(), 0)) {
					currentTree.m_isEmpty = true;
					for (int iSample = 0; iSample < numberSamples; iSample++)
						currentTree.m_sampleTreeVector[iSample].setIsEmpty(true);
				}
				currentData = null;
				currentSamplesVector = null;
			}
			index++;

		}
	}

	public void addSonOrderedByValue(ArrayList<Object[]> list, Object[] son) {
		if (list.size() == 0) {
			list.add(son);
		} else {
			int i;
			double sonValue = (double) son[3];
			for (i = 0; i < list.size(); i++) {
				double parentValue = (double) list.get(i)[3];
				if (parentValue < sonValue) {
					list.add(i, son);
					break;
				}
			}
			if (i == list.size())
				list.add(son);
		}
	}

	/**
	 * Help method for printing tree structure.
	 * 
	 * @param depth the current depth
	 * @param text  for outputting the structure
	 * @throws Exception if something goes wrong
	 */
	public void dumpTree(int depth, StringBuffer text) throws Exception {

		int i, j;

		for (i = 0; i < m_sons.length; i++) {
			text.append("\n");
			;
			for (j = 0; j < depth; j++) {
				text.append("|   ");
			}
			text.append("[" + m_order + "] ");
			text.append(m_localModel.leftSide(m_train));
			text.append(m_localModel.rightSide(i, m_train));
			if (m_sons[i].isLeaf()) {
				text.append(": ");
				text.append("[" + ((C45ItPartiallyConsolidatedPruneableClassifierTree)m_sons[i]).m_order + "] ");
				text.append(m_localModel.dumpLabel(i, m_train));
			} else {
				m_sons[i].dumpTree(depth + 1, text);
			}
		}
	}

	/**
	 * @return the m_elapsedTimeTrainingWholeCT
	 */
	public double getElapsedTimeTrainingWholeCT() {
		return m_elapsedTimeTrainingWholeCT;
	}

	/**
	 * @return the m_elapsedTimeTrainingPartialCT
	 */
	public double getElapsedTimeTrainingPartialCT() {
		return m_elapsedTimeTrainingPartialCT;
	}

	/**
	 * @return the m_elapsedTimeTrainingAssocBagging
	 */
	public double getElapsedTimeTrainingAssocBagging() {
		return m_elapsedTimeTrainingAssocBagging;
	}
}
