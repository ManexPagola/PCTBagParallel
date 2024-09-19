package weka.classifiers.trees.j48Consolidated;

import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.ClassifierSplitModel;
import weka.classifiers.trees.j48.Distribution;
import weka.core.Instances;

/**
 * Class for handling a distribution of class values based on a consolidation process.
 * *************************************************************************************
 *
 * @author Jes&uacute;s M. P&eacute;rez (txus.perez@ehu.eus) 
 * @version $Revision: 1.1 $
 */
public class DistributionConsolidated extends Distribution {

	/** for serialization */
	private static final long serialVersionUID = -6386302948424098805L;

	/**
	 * Creates a distribution with only one bag according
	 * to the vector of samples by calculating the average of the distributions.
	 * 
	 * @param samplesVector the vector of samples used for consolidation
	 */
	public DistributionConsolidated(Instances[] samplesVector) throws Exception {
		// Create the distribution object
		super(1, samplesVector[0].numClasses());
		int numberSamples = samplesVector.length;
		
		DistributionConsolidated[] distributionVector = new DistributionConsolidated[numberSamples];
		// Create the distribution related to each sample
		for(int iSample = 0; iSample < numberSamples; iSample++)
			distributionVector[iSample] = new DistributionConsolidated(samplesVector[iSample]);
		calculateMeanDistribution(distributionVector);
	}
	
	/**
	 * Creates a distribution by calculating the average of the distributions according
	 *  to each sample and given split model.
	 *
	 * @param samplesVector the vector of samples used for consolidation
	 * @param modelToUse the split model to be used to split each sample 
	 */
	public DistributionConsolidated(Instances[] samplesVector, 
		      ClassifierSplitModel modelToUse) throws Exception {
		// Create the distribution object
		super(modelToUse.numSubsets(), samplesVector[0].numClasses());
		int numberSamples = samplesVector.length;
		int attIndex = ((C45Split)modelToUse).attIndex();

		/** Vector storing the distribution according to each sample */
		DistributionConsolidated[] distributionVector = new DistributionConsolidated[numberSamples];
		// Create the distribution related to each sample using the given split model
		for(int iSample = 0; iSample < numberSamples; iSample++){
			// Only Instances with known values are relevant.
			Instances sampleWithoutMissing = samplesVector[iSample];
			sampleWithoutMissing.deleteWithMissing(attIndex);
			distributionVector[iSample] = new DistributionConsolidated(sampleWithoutMissing, modelToUse);
			// Add all Instances with unknown values for the corresponding
			// attribute to the distribution for the model, so that
			// the complete distribution is stored with the model. 
			distributionVector[iSample].addInstWithUnknown(samplesVector[iSample], attIndex);
		}
		calculateMeanDistribution(distributionVector);
	}

	/**
	 * Constructor calling the constructor of the superclass
	 * (No necessary if the above methods are moved to the official class 'Distribution')
	 *
	 * @param instances instances to be taken in account
	 */
	public DistributionConsolidated(Instances instances) throws Exception {
		super(instances);
	}

	/**
	 * Constructor calling the constructor of the superclass
	 * (No necessary if the above methods are moved to the official class 'Distribution')
	 *
	 * @param instances instances to be taken in account
	 * @param modelToUse the split model to be used to split each sample 
	 */
	public DistributionConsolidated(Instances instances, ClassifierSplitModel modelToUse) throws Exception {
		super(instances, modelToUse);
	}

	/**
	 * Calculates the distribution by calculating the average of the distributions.
	 * 
	 * @param distributionVector vector storing the distributions according to a set of samples
	 */
	private void calculateMeanDistribution(DistributionConsolidated[] distributionVector){
		int numberSamples = distributionVector.length;
		int numberClasses = distributionVector[0].numClasses();
		// Add the distributions
		for(int iSample = 0; iSample < numberSamples; iSample++)
			add(distributionVector[iSample]);
		// Calculate the mean
		for(int iBag = 0; iBag < numBags(); iBag++){
		    m_perBag[iBag] /= numberSamples;
			for(int iClass = 0; iClass < numberClasses; iClass++)
				m_perClassPerBag[iBag][iClass] /= numberSamples;
		}
		for(int iClass = 0; iClass < numberClasses; iClass++)
			m_perClass[iClass] /= numberSamples;
	    totaL /= numberSamples;
	}
	
	/**
	 * Adds the given distribution to this one.
	 * 
	 * @param distribution distribution to be added
	 */
	private final void add(DistributionConsolidated distribution) {
		for(int iBag = 0; iBag < numBags(); iBag++)
			add(iBag, distribution.getPerClassPerBag(iBag));
	}
	
	/**
	 * Gets the weights of instances per class related to given bag.
	 * 
	 * @param bagIndex index of the bag
	 * @return the weights of instances per class
	 */
	private final double [] getPerClassPerBag(int bagIndex){
	    return m_perClassPerBag[bagIndex];
	}
}
