package weka.classifiers.trees.j48Consolidated;

import weka.classifiers.trees.j48.C45Split;
import weka.classifiers.trees.j48.Distribution;
import weka.core.Attribute;
import weka.core.Instances;

/**
 * Class implementing a C4.5-type split on a consolidated attribute based on a set of samples.
 * *************************************************************************************
 * 
 * @author Jes&uacute;s M. P&eacute;rez (txus.perez@ehu.eus) 
 * @version $Revision: 1.2 $
 */
public class C45ConsolidatedSplit extends C45Split {

	/** for serialization */
	private static final long serialVersionUID = 1174832141695586851L;

	/**
	 * Creates a split model to be used to consolidate the decision around the set of samples,
	 *  but with a null distribution
	 *   
	 * @param attIndex attribute to split on
	 * @param minNoObj minimum number of objects
	 * @param sumOfWeights sum of the weights
	 * @param useMDLcorrection whether to use MDL adjustment when finding splits on numeric attributes
	 * @param splitAtt attribute to split. Only to get information about this 
	 * @param splitPointConsolidated the split point to use to split, if numerical.
	 */
	public C45ConsolidatedSplit(int attIndex, int minNoObj, double sumOfWeights, 
			boolean useMDLcorrection, Attribute splitAtt, double splitPointConsolidated) {
		super(attIndex, minNoObj, sumOfWeights, useMDLcorrection);

		// Initialize the remaining instance variables.
		m_splitPoint = splitPointConsolidated;
		m_infoGain = 0;
		m_gainRatio = 0;
		m_distribution = null;

		// Different treatment for enumerated and numeric attributes.
	    if (splitAtt.isNominal()) {
	      m_complexityIndex = splitAtt.numValues();
	      m_index = m_complexityIndex;
	      m_numSubsets = m_complexityIndex;
	    } else {
	      m_complexityIndex = 2;
	      m_index = 2;
	      m_numSubsets = 2;
	    }
	}

	/**
	 * Creates a split model for the consolidated tree based on the consolidated decision
	 *
	 * @param attIndex attribute to split on
	 * @param minNoObj minimum number of objects
	 * @param sumOfWeights sum of the weights
	 * @param useMDLcorrection whether to use MDL adjustment when finding splits on numeric attributes
	 * @param data the training sample. Only to get information about the attributes 
	 * @param samplesVector the vector of samples used for consolidation
	 * @param splitPointConsolidated the split point to use to split, if numerical.
	 * @exception Exception if something goes wrong
	 */
	public C45ConsolidatedSplit(int attIndex, int minNoObj, double sumOfWeights, boolean useMDLcorrection,
			Instances data, Instances[] samplesVector, double splitPointConsolidated) throws Exception {
		this(attIndex, minNoObj, sumOfWeights, useMDLcorrection, data.attribute(attIndex), splitPointConsolidated);
		// Set a null model with the consolidated decision to calculate the consolidated distribution
		C45ConsolidatedSplit nullModelToConsolidate = this;
		m_distribution = new DistributionConsolidated(samplesVector, nullModelToConsolidate);
		m_infoGain = infoGainCrit.splitCritValue(m_distribution, m_sumOfWeights);
		m_gainRatio = gainRatioCrit.splitCritValue(m_distribution, m_sumOfWeights, m_infoGain);
	}

	/**
	 * Creates a split model for a base tree based on the consolidated decision
	 *
	 * @param attIndex attribute to split on
	 * @param minNoObj minimum number of objects
	 * @param sumOfWeights sum of the weights
	 * @param useMDLcorrection whether to use MDL adjustment when finding splits on numeric attributes
	 * @param data the training sample related to a base tree 
	 * @param splitPointConsolidated the split point to use to split, if numerical.
	 * @exception Exception if something goes wrong
	 */
	public C45ConsolidatedSplit(int attIndex, int minNoObj, double sumOfWeights, boolean useMDLcorrection,
			Instances data, double splitPointConsolidated) throws Exception {
		this(attIndex, minNoObj, sumOfWeights, useMDLcorrection, data.attribute(attIndex), splitPointConsolidated);
		// Set a null model with the consolidated decision to calculate the consolidated distribution
		C45ConsolidatedSplit nullModelToConsolidate = this; 
		m_distribution = new Distribution(data, nullModelToConsolidate);
		m_infoGain = infoGainCrit.splitCritValue(m_distribution, m_sumOfWeights);
		m_gainRatio = gainRatioCrit.splitCritValue(m_distribution, m_sumOfWeights, m_infoGain);
	}
}

