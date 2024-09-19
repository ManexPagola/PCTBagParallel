/**
 *
 */
package weka.classifiers.trees.j48PartiallyConsolidated;

import weka.classifiers.trees.j48Consolidated.C45ConsolidatedModelSelection;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedPruneableClassifierTree;
import weka.classifiers.trees.j48ItPartiallyConsolidated.C45ItPartiallyConsolidatedPruneableClassifierTree;

import java.util.ArrayList;
import java.util.Collections;

import weka.classifiers.trees.j48.ClassifierTree;
import weka.classifiers.trees.j48.ModelSelection;
import weka.classifiers.trees.j48.NoSplit;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class for handling a consolidated tree structure that can
 * be pruned using C4.5 procedures.
 * *************************************************************************************
 * Attention! Removed 'final' modifier from collapse() function of j48/C45PruneableClassifierTree
 *  class and from cleanup() function of j48/ClassifierTree class in order to overwrite
 *  these functions here.
 * *************************************************************************************<br/>
 *
 * @author Ander Otsoa de Alda Alzaga (ander.otsoadealda@gmail.com)
 * @author Jesús M. Pérez (txus.perez@ehu.eus)
 * @version $Revision: 0.3 $
 */
public class C45PartiallyConsolidatedPruneableClassifierTree extends
		C45ConsolidatedPruneableClassifierTree {

	/** for serialization **/
	private static final long serialVersionUID = 6410655550027990502L;

	/** Number of internal nodes of the consolidated tree to be left as consolidated based
	 *  on the given consolidation percent parameter. We will know this value, once 
	 *  we know the total number of internal nodes of the complete consolidated tree. */
	protected int m_numInternalNodesConso = -1;

	/** Whether to prune the base trees without preserving the structure of the partially
	 * consolidated tree. */
	protected boolean m_pruneBaseTreesWithoutPreservingConsolidatedStructure = false;

	/** Vector for storing the generated base decision trees
	 *  related to each sample */
	protected C45PruneableClassifierTreeExtended[] m_sampleTreeVector;

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
	 * @param numberSamples Number of Samples
	 * @param notPreservingStructure true if pruning the base trees without preserving the consolidated structure
	 * @throws Exception if something goes wrong
	 */
	public C45PartiallyConsolidatedPruneableClassifierTree(
			ModelSelection toSelectLocModel, C45ModelSelectionExtended baseModelToForceDecision,
			boolean pruneTree, float cf,
			boolean raiseTree, boolean cleanup,
			boolean collapseTree, int numberSamples,
			boolean notPreservingStructure) throws Exception {
		super(toSelectLocModel, pruneTree, cf, raiseTree, cleanup, collapseTree);
		// Initialize each base decision tree of the vector
		ModelSelection modelToConsolidate = ((C45ConsolidatedModelSelection)toSelectLocModel).getModelToConsolidate();
		m_sampleTreeVector = new C45PruneableClassifierTreeExtended[numberSamples];
		for (int iSample = 0; iSample < numberSamples; iSample++)
			m_sampleTreeVector[iSample] = new C45PruneableClassifierTreeExtended(
					modelToConsolidate,	baseModelToForceDecision, pruneTree, cf, raiseTree, cleanup, collapseTree,
					notPreservingStructure);
	}

	/**
	 * Method for building a pruneable classifier consolidated tree.
	 *
	 * @param data the data for pruning the consolidated tree
	 * @param samplesVector the vector of samples for building the consolidated tree
	 * @param consolidationPercent the value of consolidation percent
	 * @throws Exception if something goes wrong
	 */
	public void buildClassifier(Instances data, Instances[] samplesVector, float consolidationPercent) throws Exception {

		buildTree(data, samplesVector, m_subtreeRaising || !m_cleanup);
		if (m_collapseTheTree) {
			collapse();
		}
		if (m_pruneTheTree) {
			prune();
		}
		leavePartiallyConsolidated(consolidationPercent);
		applyBagging();
		
		if (m_cleanup)
			cleanup(new Instances(data, 0));
	}
	
	/**
	 * Collapses a tree to a node if training error doesn't increase.
	 * And also each tree of a tree vector to a node.
	 *
	 */
	public final void collapse(){
	
		double errorsOfSubtree;
		double errorsOfTree;
		int i;
	
		if (!m_isLeaf){
			errorsOfSubtree = getTrainingErrors();
			errorsOfTree = localModel().distribution().numIncorrect();
			if (errorsOfSubtree >= errorsOfTree-1E-3)
				setAsLeaf();
			else
				for (i=0;i<m_sons.length;i++)
					son(i).collapse();
		}
	}
	
	/**
	 * Prunes the consolidated tree using C4.5's pruning procedure, and all base trees in the same way
	 *
	 * @throws Exception if something goes wrong
	 */
	public void prune() throws Exception {

		double errorsLargestBranch;
		double errorsLeaf;
		double errorsTree;
		int indexOfLargestBranch;
		C45PartiallyConsolidatedPruneableClassifierTree largestBranch;
		int i;

		if (!m_isLeaf){

			// Prune all subtrees.
			for (i=0;i<m_sons.length;i++)
				son(i).prune();

			// Compute error for largest branch
			indexOfLargestBranch = localModel().distribution().maxBag();
			if (m_subtreeRaising) {
				errorsLargestBranch = ((C45PartiallyConsolidatedPruneableClassifierTree)(son(indexOfLargestBranch))).
						getEstimatedErrorsForBranch((Instances)m_train);
			} else {
				errorsLargestBranch = Double.MAX_VALUE;
			}

			// Compute error if this Tree would be leaf
			errorsLeaf = 
					getEstimatedErrorsForDistribution(localModel().distribution());

			// Compute error for the whole subtree
			errorsTree = getEstimatedErrors();

			// Decide if leaf is best choice.
			if (Utils.smOrEq(errorsLeaf,errorsTree+0.1) &&
					Utils.smOrEq(errorsLeaf,errorsLargestBranch+0.1)){
				setAsLeaf();
				return;
			}

			// Decide if largest branch is better choice
			// than whole subtree.
			if (Utils.smOrEq(errorsLargestBranch,errorsTree+0.1)){
				largestBranch = (C45PartiallyConsolidatedPruneableClassifierTree)son(indexOfLargestBranch);
				m_sons = largestBranch.m_sons;
				m_localModel = largestBranch.localModel();
				m_isLeaf = largestBranch.m_isLeaf;
				newDistribution(m_train);
				// Replace current node with the largest branch in all base trees
				for (int iSample=0; iSample < m_sampleTreeVector.length; iSample++)
					m_sampleTreeVector[iSample].replaceWithIthSubtree(indexOfLargestBranch);
				prune();
			}
		}
	}
	  
	/**
	 * Cleanup in order to save memory.
	 * 
	 * @param justHeaderInfo
	 */
	public final void cleanup(Instances justHeaderInfo) {
		super.cleanup(justHeaderInfo);
		for (int iSample=0; iSample < m_sampleTreeVector.length; iSample++)
			m_sampleTreeVector[iSample].cleanup(justHeaderInfo);
	}
	
	/**
	 * Set current node as leaf. Also for all base trees
	 */
	private void setAsLeaf(){
		// Free adjacent trees
		m_sons = null;
		m_isLeaf = true;
	
		// Get NoSplit Model for tree.
		m_localModel = new NoSplit(localModel().distribution());
	
		// Set node as leaf in all base trees
		for (int iSample=0; iSample < m_sampleTreeVector.length; iSample++)
			m_sampleTreeVector[iSample].setAsLeaf();
	}

	/**
	 * Returns a newly created tree.
	 *
	 * @param data the data to work with
	 * @param samplesVector the vector of samples for building the consolidated tree
	 * @return the new consolidated tree
	 * @throws Exception if something goes wrong
	 */
	protected ClassifierTree getNewTree(Instances data, Instances[] samplesVector,
			ClassifierTree[] sampleTreeVectorParent, int iSon) throws Exception {
		/** Number of Samples. */
		int numberSamples = samplesVector.length;

		C45ModelSelectionExtended baseModelToForceDecision = m_sampleTreeVector[0].getBaseModelToForceDecision();
		C45PartiallyConsolidatedPruneableClassifierTree newTree =
				new C45PartiallyConsolidatedPruneableClassifierTree(m_toSelectModel, baseModelToForceDecision,
						m_pruneTheTree, m_CF, m_subtreeRaising, m_cleanup, m_collapseTheTree , samplesVector.length,
						m_pruneBaseTreesWithoutPreservingConsolidatedStructure);
		/** Set the recent created base trees like the sons of the given parent node */
		for (int iSample = 0; iSample < numberSamples; iSample++)
			((C45PruneableClassifierTreeExtended)sampleTreeVectorParent[iSample]).setIthSon(iSon, newTree.m_sampleTreeVector[iSample]);
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

		/** Select the best model to split (if it is worth) based on the consolidation proccess */
		m_localModel = ((C45ConsolidatedModelSelection)m_toSelectModel).selectModel(data, samplesVector);
		for (int iSample = 0; iSample < numberSamples; iSample++)
			m_sampleTreeVector[iSample].setLocalModel(samplesVector[iSample],m_localModel);

		if (m_localModel.numSubsets() > 1) {
			/** Vector storing the obtained subsamples after the split of data */
			Instances [] localInstances;
			/** Vector storing the obtained subsamples after the split of each sample of the vector */
			ArrayList<Instances[]> localInstancesVector = new ArrayList<Instances[]>();
			
			/** For some base trees, although the current node is not a leaf, it could be empty.
			 *  This is necessary in order to calculate correctly the class membership probabilities
			 *   for the given test instance in each base tree */
			for (int iSample = 0; iSample < numberSamples; iSample++)
				if (Utils.eq(m_sampleTreeVector[iSample].getLocalModel().distribution().total(), 0))
					m_sampleTreeVector[iSample].setIsEmpty(true);
			
			/** Split data according to the consolidated m_localModel */
			localInstances = m_localModel.split(data);
			for (int iSample = 0; iSample < numberSamples; iSample++)
				localInstancesVector.add(m_localModel.split(samplesVector[iSample]));

			/** Create the child nodes of the current node and call recursively to getNewTree() */
			data = null;
			samplesVector = null;
			m_sons = new ClassifierTree [m_localModel.numSubsets()];
			for (int iSample = 0; iSample < numberSamples; iSample++)
				((C45PruneableClassifierTreeExtended)m_sampleTreeVector[iSample]).createSonsVector(m_localModel.numSubsets());
			for (int iSon = 0; iSon < m_sons.length; iSon++) {
				/** Vector storing the subsamples related to the iSon-th son */
				Instances[] localSamplesVector = new Instances[numberSamples];
				for (int iSample = 0; iSample < numberSamples; iSample++)
					localSamplesVector[iSample] =
						((Instances[]) localInstancesVector.get(iSample))[iSon];
				m_sons[iSon] = (C45PartiallyConsolidatedPruneableClassifierTree)getNewTree(
									localInstances[iSon], localSamplesVector, m_sampleTreeVector, iSon);

				localInstances[iSon] = null;
				localSamplesVector = null;
			}
			localInstances = null;
			localInstancesVector.clear();
		}else{
			m_isLeaf = true;
			for (int iSample = 0; iSample < numberSamples; iSample++)
				m_sampleTreeVector[iSample].setIsLeaf(true);

			if (Utils.eq(m_localModel.distribution().total(), 0)){
				m_isEmpty = true;
				for (int iSample = 0; iSample < numberSamples; iSample++)
					m_sampleTreeVector[iSample].setIsEmpty(true);
			}
			data = null;
			samplesVector = null;
		}
	}

	/**
	 * Getter for m_sampleTreeVector member.
	 *
	 * @return m_sampleTreeVector Vector for storing the generated base decision trees
	 *  related to each sample
	 */
	public ClassifierTree[] getSampleTreeVector() {
		return m_sampleTreeVector;
	}

	/**
	 * Returns the i-th base decision tree of the vector.
	 *
	 * @param iSample Index of the base decision tree
	 * @return the i-th base decision tree
	 */
	public ClassifierTree getSampleTreeIth(int iSample) {
		return m_sampleTreeVector[iSample];
	}

	/**
	 * @return the m_numInternalNodesConso
	 */
	public int getNumInternalNodesConso() {
		return m_numInternalNodesConso;
	}

	/**
	 * @param m_numInternalNodesConso the m_numInternalNodesConso to set
	 */
	protected void setNumInternalNodesConso(int numNodes) {
		m_numInternalNodesConso = numNodes;
	}

	/**
	 * Prunes the consolidated tree and all base trees according consolidationPercent.
	 *
	 * @param consolidationPercent percentage of the structure of the tree to leave without pruning 
	 */
	public void leavePartiallyConsolidated(float consolidationPercent) {
		// Number of internal nodes of the consolidated tree
		int innerNodes = numNodes() - numLeaves();
		// Number of nodes of the consolidated tree to leave as consolidated based on given consolidationPercent 
		int numberNodesConso = (int)(((innerNodes * consolidationPercent) / 100) + 0.5);
		setNumInternalNodesConso(numberNodesConso);
		System.out.println("Number of nodes to leave as consolidated: " + numberNodesConso + " of " + innerNodes);
		// Vector storing the nodes to maintain as consolidated
		ArrayList<C45PartiallyConsolidatedPruneableClassifierTree> nodesConsoVector = new ArrayList<C45PartiallyConsolidatedPruneableClassifierTree>();
		// Vector storing the weight of the nodes of nodesConsoVector
		ArrayList<Double> weightNodesConsoVector = new ArrayList<Double>(); 
		// Counter of the current number of nodes left as consolidated
		int countNodesConso = 0;
		
		/** Initialize the vectors with the root node (if it has children) */
		if(!m_isLeaf){
			nodesConsoVector.add(this);
			weightNodesConsoVector.add(localModel().distribution().total());
		}
		/** Determine which nodes will be left as consolidated according to their weight 
		 *   starting from the root node */
		while((nodesConsoVector.size() > 0) && (countNodesConso < numberNodesConso)){
			/** Add the heaviest node */
			// Look for the heaviest node
			Double valHeaviest = Collections.max(weightNodesConsoVector);
			int iHeaviest = weightNodesConsoVector.indexOf(valHeaviest);
			C45PartiallyConsolidatedPruneableClassifierTree heaviestNode = nodesConsoVector.get(iHeaviest); 
			// Add the children of the chosen node to the vectors (ONLY if each child is an internal node)
			// // By construction it's guaranteed that heaviestNode has children
			for(int iSon = 0; iSon < heaviestNode.m_sons.length; iSon++)
				if(!(((C45PartiallyConsolidatedPruneableClassifierTree)heaviestNode.son(iSon)).m_isLeaf)){
					C45PartiallyConsolidatedPruneableClassifierTree localSon = (C45PartiallyConsolidatedPruneableClassifierTree)heaviestNode.son(iSon); 
					nodesConsoVector.add(localSon);
					weightNodesConsoVector.add(localSon.localModel().distribution().total());
				}
			// Remove the heaviest node of the vectors
			nodesConsoVector.remove(iHeaviest);
			weightNodesConsoVector.remove(iHeaviest);
			// Increase the counter of consolidated nodes
			countNodesConso++;
		}
		/** Prune the rest of nodes (also on the base trees)*/
		for(int iNode = 0; iNode < nodesConsoVector.size(); iNode++)
			((C45PartiallyConsolidatedPruneableClassifierTree)nodesConsoVector.get(iNode)).setAsLeaf();
	}

	/**
	 * Rebuilds each base tree according to J48 algorithm independently and
	 *  maintaining the consolidated tree structure
	 * @throws Exception if something goes wrong
	 */
	protected void applyBagging() throws Exception {
		/** Number of Samples. */
		int numberSamples = m_sampleTreeVector.length;
		for (int iSample = 0; iSample < numberSamples; iSample++)
			m_sampleTreeVector[iSample].rebuildTreeFromConsolidatedStructureAndPrune();
	}
	
	/**
	 * Returns number of levels in tree structure.
	 * 
	 * @return the number of levels
	 */

	public int numLevels() {
		if (m_isLeaf) {
			return 0;
		} else {
			int maxLevels = -1;
			for (int i = 0; i < m_sons.length; i++) {
				int nl = ((C45PartiallyConsolidatedPruneableClassifierTree) m_sons[i]).numLevels();
				if (nl > maxLevels) {
					maxLevels = nl;

				}
			}
			return 1 + maxLevels;
		}
	}
}