package weka.classifiers.trees.j48Consolidated;

import weka.classifiers.trees.j48.*;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.matrix.DoubleVector;

/**
 * Class for selecting a C4.5Consolidated-type split for a given dataset.
 * *************************************************************************************<br/>
 * 
 * @author Jes&uacute;s M. P&eacute;rez (txus.perez@ehu.eus) 
 * @version $Revision: 1.3 $
 */
public class C45ConsolidatedModelSelection extends C45ModelSelection {

	/** for serialization */
	private static final long serialVersionUID = 970984256023901098L;

	/** The model selection method to consolidate. */  
	protected ModelSelection m_toSelectModelToConsolidate;

	/**
	 * Initializes the split selection method with the given parameters.
	 * At the moment, only accepted C45ModelSelection
	 *
	 * @param minNoObj minimum number of instances that have to occur in at least
	 *          two subsets induced by split
	 * @param allData FULL training dataset (necessary for selection of split
	 *          points).
	 * @param useMDLcorrection whether to use MDL adjustement when finding splits
	 *          on numeric attributes
	 * @param doNotMakeSplitPointActualValue if true, split point is not relocated
	 *          by scanning the entire dataset for the closest data value
	 */
	public C45ConsolidatedModelSelection(int minNoObj, Instances allData,
			boolean useMDLcorrection, boolean doNotMakeSplitPointActualValue) {
		super(minNoObj, allData, useMDLcorrection, doNotMakeSplitPointActualValue);

		m_toSelectModelToConsolidate = new C45ModelSelection(minNoObj, allData,
				useMDLcorrection, doNotMakeSplitPointActualValue); 
	}

	/**
	 * Getter of m_toSelectModelToConsolidate
	 * @return the m_toSelectModelToConsolidate
	 */
	public ModelSelection getModelToConsolidate() {
		return m_toSelectModelToConsolidate;
	}

	/**
	 * Selects Consolidated-type split based on C4.5 for the given dataset.
	 * 
	 * @param data the data to train the classifier with
	 * @param samplesVector the vector of samples
	 * @return the consolidated model to be used to split
	 * @throws Exception  if something goes wrong
	 */
	public ClassifierSplitModel selectModel(Instances data, Instances[] samplesVector) throws Exception{

		/** Number of Samples. */
		int numberSamples = samplesVector.length;
		/** Vector storing the chosen attribute to split in each sample */
		int[] attIndexVector = new int[numberSamples];
		/** Vector storing the split point to use to split, if numerical, in each sample */
		double[] splitPointVector = new double[numberSamples];

		// Select C4.5-type split for each sample and
		//  save the chosen attribute (and the split point if numerical) to split 
		for (int iSample = 0; iSample < numberSamples; iSample++) {
			ClassifierSplitModel localModel = m_toSelectModelToConsolidate.selectModel(samplesVector[iSample]);
			if(localModel.numSubsets() > 1){
				attIndexVector[iSample] = ((C45Split) localModel).attIndex();
				splitPointVector[iSample] = ((C45Split) localModel).splitPoint();
			}else{
				attIndexVector[iSample] = -1;
				splitPointVector[iSample] = -1;
			}
		}
		// Get the most voted attribute (index)
		int votesCountByAtt[] = new int[data.numAttributes()];
		int numberVotes = 0;
		for (int iAtt = 0; iAtt < data.numAttributes(); iAtt++)
			votesCountByAtt[iAtt] = 0;
		for (int iSample = 0; iSample < numberSamples; iSample++)
			if(attIndexVector[iSample]!=-1){
				votesCountByAtt[attIndexVector[iSample]]++;
				numberVotes++;
			}
		int mostVotedAtt = Utils.maxIndex(votesCountByAtt);

		Distribution checkDistribution = new DistributionConsolidated(samplesVector);
		NoSplit noSplitModel = new NoSplit(checkDistribution);
		// if all nodes are leafs,
		if(numberVotes==0)
			//  return a consolidated leaf
			return noSplitModel;
		
		// Consolidate the split point (if numerical)
		double splitPointConsolidated = consolidateSplitPoint(mostVotedAtt, attIndexVector, splitPointVector, data);
		// Creates the consolidated model
		C45ConsolidatedSplit consolidatedModel =
				new C45ConsolidatedSplit(mostVotedAtt, m_minNoObj, checkDistribution.total(), 
						m_useMDLcorrection, data, samplesVector, splitPointConsolidated);

//		// Set the split point analogue to C45 if attribute numeric.
//		// // It is not necessary for the consolidation process because the median value 
//		// //  is already one of the proposed split points.
//		consolidatedModel.setSplitPoint(data);
		
		if(!consolidatedModel.checkModel())
			return noSplitModel;
		return consolidatedModel;
	}

	/**
	 * Calculates the median of the split points related to 'mostVotedAtt' attribute, if this is numerical
	 *  (MAX_VALUE otherwise).
	 * 
	 * @param mostVotedAtt the most voted attribute (index)
	 * @param attIndexVector Vector storing the chosen attribute to split in each sample
	 * @param splitPointVector Vector storing the split point to use to split, if numerical, in each sample
	 * @param data the training sample. Only to know if mostVotedAtt is numerical
	 * @return the consolidated split point 
	 */
	protected double consolidateSplitPoint(int mostVotedAtt, 
			int[] attIndexVector, double[] splitPointVector, Instances data){
		int numberSamples = attIndexVector.length;
		double consolidatedSplitPoint = Double.MAX_VALUE;
		if(data.attribute(mostVotedAtt).isNumeric()){
			DoubleVector splitPointChosenAttVector = new DoubleVector();
			for (int iSample = 0; iSample < numberSamples; iSample++)
				if(attIndexVector[iSample] == mostVotedAtt)
					splitPointChosenAttVector.addElement(splitPointVector[iSample]);
			// Number of split points related to chosen attribute
			int numberSplitPoints = splitPointChosenAttVector.size();
			// Get the median of the split points vector
			// // TODO median could be a method to insert in the class 'DoubleVector'
			splitPointChosenAttVector.sort();
			consolidatedSplitPoint = splitPointChosenAttVector.get(((numberSplitPoints+1)/2)-1);
		}
		return consolidatedSplitPoint;
	}

}
