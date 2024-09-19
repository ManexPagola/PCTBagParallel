package weka.classifiers.trees.j48PartiallyConsolidated;

import weka.classifiers.trees.j48.C45ModelSelection;
import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.Distribution;
import weka.classifiers.trees.j48.NoSplit;
import weka.classifiers.trees.j48Consolidated.C45ConsolidatedSplit;
import weka.core.Instances;

/**
 * Class for extend handling C45ModelSelection class
 * *************************************************************************************
 *
 * @author Ander Otsoa de Alda Alzaga (ander.otsoadealda@gmail.com)
 * @author Jesús M. Pérez (txus.perez@ehu.eus)
 * @version $Revision: 0.3 $
 */
public class C45ModelSelectionExtended extends C45ModelSelection {

	/** for serialization */
	private static final long serialVersionUID = 690999593328557886L;

	/**
	 * Constructor. 
	 * @param minNoObj minimum number of instances that have to occur in at least
	 *          two subsets induced by split
	 * @param allData FULL training dataset (necessary for selection of split
	 *          points).
	 * @param useMDLcorrection whether to use MDL adjustement when finding splits
	 *          on numeric attributes
	 * @param doNotMakeSplitPointActualValue if true, split point is not relocated
	 *          by scanning the entire dataset for the closest data value
	 */
	public C45ModelSelectionExtended(int minNoObj, Instances allData,
			boolean useMDLcorrection, boolean doNotMakeSplitPointActualValue) {
		super(minNoObj, allData, useMDLcorrection, doNotMakeSplitPointActualValue);
	}

	/**
	 * Set m_localModel based on the consolidated model taking into account the sample. 
	 * @param data instances in the current node related to the corresponding base decision tree
	 * @param consolidatedModel is the consolidated split
	 * @throws Exception if something goes wrong
	 */
	public ClassifierSplitModel selectModel(Instances data,
			ClassifierSplitModel consolidatedModel) throws Exception {
	    Distribution checkDistribution;
	    NoSplit noSplitModel = null;

		checkDistribution = new Distribution(data);
	    noSplitModel = new NoSplit(checkDistribution);
	    if(consolidatedModel.numSubsets() == 1)
	    	return noSplitModel;
	    
		// Creates the local model based on the consolidated model
		C45ConsolidatedSplit localModel =
				new C45ConsolidatedSplit(
						((C45Split) consolidatedModel).attIndex(), m_minNoObj, checkDistribution.total(),
						m_useMDLcorrection , data, ((C45Split) consolidatedModel).splitPoint());

//		// Set the split point analogue to C45 if attribute numeric.
//		// // It is not necessary for the consolidation process because the median value 
//		// //  is already one of the proposed split points.
//		localModel.setSplitPoint(data);
		
		if(!localModel.checkModel())
			return noSplitModel;
		return localModel;
	}
}