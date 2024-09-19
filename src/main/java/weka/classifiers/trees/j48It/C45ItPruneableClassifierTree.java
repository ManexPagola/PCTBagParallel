package weka.classifiers.trees.j48It;

import java.util.ArrayList;

import weka.classifiers.trees.J48It;
import weka.classifiers.trees.j48.C45PruneableClassifierTree;
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
 * 
 * @author Jes&uacute;s M. P&eacute;rez (txus.perez@ehu.eus)
 * @version $Revision: 1.2 $
 */
public class C45ItPruneableClassifierTree extends C45PruneableClassifierTree {

	/** for serialization */
	private static final long serialVersionUID = 2660972525647728377L;

	/** Indicates the order in which the node was treated */
	private int m_order;

	/**
	 * Builds the tree up to a maximum of depth levels. Set m_maximumLevel to 0 for
	 * default.
	 */
	private int m_maximumCriteria;

	/** Indicates the criteria that should be used to build the tree */
	private int m_priorityCriteria;


	/**
	 * Constructor for pruneable consolidated tree structure. Calls the superclass
	 * constructor.
	 *
	 * @param toSelectLocModel 		selection method for local splitting model
	 * @param pruneTree        		true if the tree is to be pruned
	 * @param cf               		the confidence factor for pruning
	 * @param raiseTree        		true if subtree raising has to be performed
	 * @param cleanup          		true if cleanup has to be done
	 * @param collapseTree     		true if collapse has to be done
	 * @param ITmaximumCriteria 	maximum number of nodes or levels
	 * @param ITpriorityCriteria 	criteria to build the tree
	 * @throws Exception if something goes wrong
	 */
	public C45ItPruneableClassifierTree(ModelSelection toSelectLocModel, boolean pruneTree, float cf, boolean raiseTree,
			boolean cleanup, boolean collapseTree, int ITmaximumCriteria, int ITpriorityCriteria) throws Exception {
		super(toSelectLocModel, pruneTree, cf, raiseTree, cleanup, collapseTree);
		m_maximumCriteria = ITmaximumCriteria;
		m_priorityCriteria = ITpriorityCriteria;
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
	public void buildTree(Instances data, boolean keepData) throws Exception {

		ArrayList<Object[]> list = new ArrayList<>();

						 // add(Data, tree, orderValue, currentLevel)
		list.add(new Object[] { data, this, null, 0}); // The parent node is considered level 0

		Instances[] localInstances;

		int index = 0;
		double orderValue;
		
		int internalNodes = 0;

		while (list.size() > 0) {
			Object[] current = list.get(0);
			int currentLevel = (int) current[3];

			list.set(0, null); // Null to free up memory
			list.remove(0);

			Instances currentData = (Instances) current[0];
			C45ItPruneableClassifierTree currentTree = (C45ItPruneableClassifierTree) current[1];
			currentTree.m_order = index;

			if (keepData) {
				currentTree.m_train = currentData;
			}
			currentTree.m_test = null;
			currentTree.m_isLeaf = false;
			currentTree.m_isEmpty = false;
			currentTree.m_sons = null;
			currentTree.m_localModel = currentTree.m_toSelectModel.selectModel(currentData);

			if ((currentTree.m_localModel.numSubsets() > 1) && ((m_priorityCriteria == J48It.Original)
					|| ((m_priorityCriteria == J48It.Levelbylevel) && (currentLevel < m_maximumCriteria))
					|| ((m_priorityCriteria > J48It.Levelbylevel) && (internalNodes < m_maximumCriteria)))) {


				ArrayList<Object[]> listSons = new ArrayList<>();
				localInstances = currentTree.m_localModel.split(currentData);
				currentData = null;
				currentTree.m_sons = new ClassifierTree[currentTree.m_localModel.numSubsets()];
				for (int i = 0; i < currentTree.m_sons.length; i++) {
					ClassifierTree newTree = new C45ItPruneableClassifierTree(currentTree.m_toSelectModel,
							m_pruneTheTree, m_CF, m_subtreeRaising, m_cleanup, m_collapseTheTree, m_maximumCriteria,
							m_priorityCriteria);
					
					if (m_priorityCriteria == J48It.Size) // Added by size, largest to smallest
					{

						orderValue = currentTree.m_localModel.distribution().perBag(i);

						Object[] son = new Object[] { localInstances[i], newTree, orderValue, currentLevel + 1 };
						addSonOrderedByValue(list, son);
					} else if (m_priorityCriteria == J48It.Gainratio) // Added by gainratio, largest to smallest
					{
						ClassifierSplitModel sonModel = ((C45ItPruneableClassifierTree) newTree).m_toSelectModel
								.selectModel(localInstances[i]);
						if (sonModel.numSubsets() > 1) {

							orderValue = ((C45Split) sonModel).gainRatio();

						} else {

							orderValue = (double) Double.MIN_VALUE;
						}
						Object[] son = new Object[] { localInstances[i], newTree, orderValue, currentLevel + 1 };
						addSonOrderedByValue(list, son);
					} else if (m_priorityCriteria == J48It.Gainratio_normalized) // Added by gainratio normalized,
					// largest to smallest
					{

						double size = currentTree.m_localModel.distribution().perBag(i);
						double gainRatio;
						ClassifierSplitModel sonModel = ((C45ItPruneableClassifierTree) newTree).m_toSelectModel
								.selectModel(localInstances[i]);
						if (sonModel.numSubsets() > 1) {

							gainRatio = ((C45Split) sonModel).gainRatio();
							orderValue = size * gainRatio;

						} else {

							orderValue = (double) Double.MIN_VALUE;
						}
						Object[] son = new Object[] { localInstances[i], newTree, orderValue, currentLevel + 1 };
						addSonOrderedByValue(list, son);

					} else {
						listSons.add(new Object[] { localInstances[i], newTree, 0, currentLevel + 1 });
					}

					currentTree.m_sons[i] = newTree;

					localInstances[i] = null;
				}

				if (m_priorityCriteria == J48It.Levelbylevel) { // Level by level
					list.addAll(listSons);
				}

				else if (m_priorityCriteria == J48It.Preorder || m_priorityCriteria == J48It.Original) { // Preorder
					listSons.addAll(list);
					list = listSons;
				}

				listSons = null;
				internalNodes ++;
				
			} else {
				currentTree.m_isLeaf = true;
				if (Utils.eq(currentData.sumOfWeights(), 0)) {
					currentTree.m_isEmpty = true;
				}
				currentData = null;
			}
			index++;
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
				text.append("[" + ((C45ItPruneableClassifierTree) m_sons[i]).m_order + "] ");
				text.append(m_localModel.dumpLabel(i, m_train));
			} else {
				m_sons[i].dumpTree(depth + 1, text);
			}
		}
	}

	public void addSonOrderedByValue(ArrayList<Object[]> list, Object[] son) {
		if (list.size() == 0) {
			list.add(0, son);
		} else {

			double sonValue = (double) son[2];
			for (int i = 0; i < list.size(); i++) {

				double parentValue = (double) list.get(i)[2];
				if (parentValue < sonValue) {
					list.add(i, son);
					break;
				}

				if (i == list.size() - 1) {
					list.add(son);
					break;
				}

			}
		}

	}

}