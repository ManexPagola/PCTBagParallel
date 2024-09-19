/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    ZeroR.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.classifiers.rules;

import java.util.Enumeration;
import java.util.Vector;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.AttributeStats;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.Summarizable;
import weka.core.Utils;
import weka.core.WeightedInstancesHandler;

/**
 * <!-- globalinfo-start --> Class for building and using a 0-R classifier.
 * Predicts the mean (for a numeric class) or the mode (for a nominal class).
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 12024 $
 */
public class ZeroR extends AbstractClassifier implements
WeightedInstancesHandler, Sourcable, AdditionalMeasureProducer, Summarizable {

	/** for serialization */
	static final long serialVersionUID = 48055541465867954L;

	/** The class value 0R predicts. */
	private double m_ClassValue;

	/** The number of instances in each class (null if class numeric). */
	private double[] m_Counts;

	/** The class attribute. */
	private Attribute m_Class;

	private Instances m_data;

	/**
	 * Returns a string describing classifier
	 * 
	 * @return a description suitable for displaying in the explorer/experimenter
	 *         gui
	 */
	public String globalInfo() {
		return "Class for building and using a 0-R classifier. Predicts the mean "
				+ "(for a numeric class) or the mode (for a nominal class).";
	}

	/**
	 * Returns default capabilities of the classifier.
	 * 
	 * @return the capabilities of this classifier
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.STRING_ATTRIBUTES);
		result.enable(Capability.RELATIONAL_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);
		result.enable(Capability.DATE_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		// instances
		result.setMinimumNumberInstances(0);

		return result;
	}

	/**
	 * Generates the classifier.
	 * 
	 * @param instances set of instances serving as training data
	 * @throws Exception if the classifier has not been generated successfully
	 */
	@Override
	public void buildClassifier(Instances instances) throws Exception {
		// can classifier handle the data?
		getCapabilities().testWithFail(instances);

		m_data = instances;

		double sumOfWeights = 0;

		m_Class = instances.classAttribute();
		m_ClassValue = 0;
		switch (instances.classAttribute().type()) {
		case Attribute.NUMERIC:
			m_Counts = null;
			break;
		case Attribute.NOMINAL:
			m_Counts = new double[instances.numClasses()];
			for (int i = 0; i < m_Counts.length; i++) {
				m_Counts[i] = 1;
			}
			sumOfWeights = instances.numClasses();
			break;
		}
		for (Instance instance : instances) {
			double classValue = instance.classValue();
			if (!Utils.isMissingValue(classValue)) {
				if (instances.classAttribute().isNominal()) {
					m_Counts[(int) classValue] += instance.weight();
				} else {
					m_ClassValue += instance.weight() * classValue;
				}
				sumOfWeights += instance.weight();
			}
		}
		if (instances.classAttribute().isNumeric()) {
			if (Utils.gr(sumOfWeights, 0)) {
				m_ClassValue /= sumOfWeights;
			}
		} else {
			m_ClassValue = Utils.maxIndex(m_Counts);
			Utils.normalize(m_Counts, sumOfWeights);
		}
	}

	/**
	 * Classifies a given instance.
	 * 
	 * @param instance the instance to be classified
	 * @return index of the predicted class
	 */
	@Override
	public double classifyInstance(Instance instance) {

		return m_ClassValue;
	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 * 
	 * @param instance the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception if class is numeric
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {

		if (m_Counts == null) {
			double[] result = new double[1];
			result[0] = m_ClassValue;
			return result;
		} else {
			return m_Counts.clone();
		}
	}

	/**
	 * Returns a string that describes the classifier as source. The classifier
	 * will be contained in a class with the given name (there may be auxiliary
	 * classes), and will contain a method with the signature:
	 * 
	 * <pre>
	 * <code>
	 * public static double classify(Object[] i);
	 * </code>
	 * </pre>
	 * 
	 * where the array <code>i</code> contains elements that are either Double,
	 * String, with missing values represented as null. The generated code is
	 * public domain and comes with no warranty.
	 * 
	 * @param className the name that should be given to the source class.
	 * @return the object source described by a string
	 * @throws Exception if the souce can't be computed
	 */
	@Override
	public String toSource(String className) throws Exception {
		StringBuffer result;

		result = new StringBuffer();

		result.append("class " + className + " {\n");
		result.append("  public static double classify(Object[] i) {\n");
		if (m_Counts != null) {
			result.append("    // always predicts label '"
					+ m_Class.value((int) m_ClassValue) + "'\n");
		}
		result.append("    return " + m_ClassValue + ";\n");
		result.append("  }\n");
		result.append("}\n");

		return result.toString();
	}

	/**
	 * Returns a description of the classifier.
	 * 
	 * @return a description of the classifier as a string.
	 */
	@Override
	public String toString() {

		if (m_Class == null) {
			return "ZeroR: No model built yet.";
		}
		if (m_Counts == null) {
			return "ZeroR predicts class value: " + m_ClassValue;
		} else {
			return "ZeroR predicts class value: " + m_Class.value((int) m_ClassValue);
		}
	}

	/**
	 * Returns the revision string.
	 * 
	 * @return the revision
	 */
	@Override
	public String getRevision() {
		return RevisionUtils.extract("$Revision: 12024 $");
	}

	/**
	 * Main method for testing this class.
	 * 
	 * @param argv the options
	 */
	public static void main(String[] argv) {
		runClassifier(new ZeroR(), argv);
	}

	/**
	 * Returns number of attributes of the dataset (without class). 
	 * 
	 * @return number of attributes of the dataset (without class).
	 */
	public double measureNumAttributes() {
		return (double)(m_data.numAttributes() - 1);
	}

	/**
	 * Returns number of numeric attributes of the dataset (without class). 
	 * 
	 * @return number of numeric attributes of the dataset (without class).
	 */
	public double measureNumNumericAttributes() {
		int count = 0;
		// For each attribute.
		for (int i = 0; i < m_data.numAttributes(); i++) {
			if (i == m_data.classIndex())
				continue;
			Attribute at = m_data.attribute(i);
			if (at.isNumeric())
				count++;
		}
		return (double)count;
	}

	/**
	 * Returns number of nominal attributes of the dataset (without class). 
	 * 
	 * @return number of nominal attributes of the dataset (without class).
	 */
	public double measureNumNominalAttributes() {
		int count = 0;
		// For each attribute.
		for (int i = 0; i < m_data.numAttributes(); i++) {
			if (i == m_data.classIndex())
				continue;
			Attribute at = m_data.attribute(i);
			if (at.isNominal())
				count++;
		}
		return (double)count;
	}

	/**
	 * Returns whether there are missing values in the dataset or not (without class). 
	 *  
	 * @return 1.0 if there are missing values in the dataset or 0.0, if not.
	 */
	public double measureMissingValues() {
		// For each attribute.
		for (int i = 0; i < m_data.numAttributes(); i++) {
			if (i == m_data.classIndex())
				continue;
			AttributeStats as = m_data.attributeStats(i);
			if (as.missingCount > 0)
				return (double)1.0;
		}
		return (double)0.0;
	}

	/**
	 * Returns number of attributes with missing values (without class). 
	 * 
	 * @return number of attributes with missing values (without class).
	 */
	public double measureNumAttsMissingValues() {
		int count = 0;
		// For each attribute.
		for (int i = 0; i < m_data.numAttributes(); i++) {
			if (i == m_data.classIndex())
				continue;
			AttributeStats as = m_data.attributeStats(i);
			if (as.missingCount > 0)
				count++;
		}
		return (double)count;
	}

	/**
	 * Returns number of examples with missing values in the dataset (without class). 
	 * 
	 * @return number of examples with missing values in the dataset (without class).
	 */
	public double measureNumMissingValuesDataset() {

		int sum = 0;
		// For each attribute.
		for (int i = 0; i < m_data.numAttributes(); i++) {
			if (i == m_data.classIndex())
				continue;
			AttributeStats as = m_data.attributeStats(i);
			sum += as.missingCount;
		}
		return (double)sum;
	}

	/**
	 * Returns percentage of examples with missing values in the dataset (without class). 
	 * 
	 * @return percentage of examples with missing values in the dataset (without class).
	 */
	public double measurePercentMissingValuesDataset() {

		int tamDataset = m_data.numInstances() * m_data.numAttributes();
		int sum = 0;
		// For each attribute.
		for (int i = 0; i < m_data.numAttributes(); i++) {
			if (i == m_data.classIndex())
				continue;
			AttributeStats as = m_data.attributeStats(i);
			sum += as.missingCount;
		}
		return (double)sum/tamDataset*100.0;
	}

	/**
	 * Returns number of classes in the dataset. 
	 * 
	 * @return number of classes in the dataset.
	 */
	public double measureNumClasses() {

		return (double)m_data.numClasses();
	}

	/**
	 * Returns whether there are any empty classes or not. 
	 * 
	 * @return 1.0 if there are any empty classes or 0.0, if not. 
	 * 
	 */
	public double measureEmptyClass() {

		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		int count = classCounts[Utils.minIndex(classCounts)];
		if (count == 0)
			return (double)1.0;
		else
			return (double)0.0;
	}

	/**
	 * Returns number of examples of the first class
	 * 
	 * @return number of examples of minority class.
	 */
	public double measureNumFirstClass() {

		int count=0;
		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		count = classCounts[0];
		return (double)count;
	}

	/**
	 * Returns index of minority class
	 * (discarding empty classes). 
	 * 
	 * @return index of minority class.
	 */
	public double measureMinClassIndex() {
		
		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		int i_iMinClass;
		for(i_iMinClass = 0; ((i_iMinClass < classCounts.length) && (Utils.kthSmallestValue(classCounts, i_iMinClass) == 0)); i_iMinClass++);
		if(i_iMinClass < classCounts.length)
			return (double)i_iMinClass;
		else 
			return (double)-1;
	}

	/**
	 * Returns number of examples of minority class
	 * (discarding empty classes). 
	 * 
	 * @return number of examples of minority class.
	 */
	public double measureNumMinClass() {

		int count=0;
		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		int i_iMinClass;
		for(i_iMinClass = 0; ((i_iMinClass < classCounts.length) && (Utils.kthSmallestValue(classCounts, i_iMinClass) == 0)); i_iMinClass++);
		if(i_iMinClass < classCounts.length)
			count = Utils.kthSmallestValue(classCounts,i_iMinClass);
		return (double)count;
	}

	/**
	 * Returns percentage of examples of minority class
	 * (discarding empty classes). 
	 * 
	 * @return percentage of examples of minority class.
	 */
	public double measurePercentMinClass() {

		int count=0;
		int numInstances = m_data.numInstances();
		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		int i_iMinClass;
		for(i_iMinClass = 0; ((i_iMinClass < classCounts.length) && (Utils.kthSmallestValue(classCounts, i_iMinClass) == 0)); i_iMinClass++);
		if(i_iMinClass < classCounts.length)
			count = Utils.kthSmallestValue(classCounts,i_iMinClass);
		return (double)count/numInstances*100.0;
	}

	/**
	 * Returns number of examples of majority class. 
	 * 
	 * @return number of examples of majority class.
	 */
	public double measureNumMajClass() {

		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		int count = classCounts[Utils.maxIndex(classCounts)];
		return (double)count;
	}

	/**
	 * Returns percentage of examples of majority class. 
	 * 
	 * @return percentage of examples of majority class.
	 */
	public double measurePercentMajClass() {

		int numInstances = m_data.numInstances();
		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;
		int count = classCounts[Utils.maxIndex(classCounts)];
		return (double)count/numInstances*100.0;
	}

	/**
	 * Returns an enumeration of the additional measure names
	 * 
	 * @return an enumeration of the measure names
	 */
	@Override
	public Enumeration<String> enumerateMeasures() {
		Vector<String> newVector = new Vector<String>(3);
		newVector.addElement("measureNumAttributes");
		newVector.addElement("measureNumNumericAttributes");
		newVector.addElement("measureNumNominalAttributes");
		newVector.addElement("measureMissingValues");
		newVector.addElement("measureNumAttsMissingValues");
		newVector.addElement("measureNumMissingValuesDataset");
		newVector.addElement("measurePercentMissingValuesDataset");
		newVector.addElement("measureNumClasses");
		newVector.addElement("measureEmptyClass");
		newVector.addElement("measureNumFirstClass");
		newVector.addElement("measureMinClassIndex");
		newVector.addElement("measureNumMinClass");
		newVector.addElement("measurePercentMinClass");
		newVector.addElement("measureNumMajClass");
		newVector.addElement("measurePercentMajClass");
		return newVector.elements();
	}

	/**
	 * Returns the value of the named measure
	 * 
	 * @param additionalMeasureName the name of the measure to query for its value
	 * @return the value of the named measure
	 * @throws IllegalArgumentException if the named measure is not supported
	 */
	@Override
	public double getMeasure(String additionalMeasureName) {
		if (additionalMeasureName.compareToIgnoreCase("measureNumAttributes") == 0) {
			return measureNumAttributes();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumNumericAttributes") == 0) {
			return measureNumNumericAttributes();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumNominalAttributes") == 0) {
			return measureNumNominalAttributes();
		} else if (additionalMeasureName.compareToIgnoreCase("measureMissingValues") == 0) {
			return measureMissingValues();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumAttsMissingValues") == 0) {
			return measureNumAttsMissingValues();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumMissingValuesDataset") == 0) {
			return measureNumMissingValuesDataset();
		} else if (additionalMeasureName.compareToIgnoreCase("measurePercentMissingValuesDataset") == 0) {
			return measurePercentMissingValuesDataset();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumClasses") == 0) {
			return measureNumClasses();
		} else if (additionalMeasureName.compareToIgnoreCase("measureEmptyClass") == 0) {
			return measureEmptyClass();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumFirstClass") == 0) {
			return measureNumFirstClass();
		} else if (additionalMeasureName.compareToIgnoreCase("measureMinClassIndex") == 0) {
			return measureMinClassIndex();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumMinClass") == 0) {
			return measureNumMinClass();
		} else if (additionalMeasureName.compareToIgnoreCase("measurePercentMinClass") == 0) {
			return measurePercentMinClass();
		} else if (additionalMeasureName.compareToIgnoreCase("measureNumMajClass") == 0) {
			return measureNumMajClass();
		} else if (additionalMeasureName.compareToIgnoreCase("measurePercentMajClass") == 0) {
			return measurePercentMajClass();
		} else {
			throw new IllegalArgumentException(additionalMeasureName
					+ " not supported (ZeroR)");
		}
	}

	/**
	 * Returns a superconcise version of the model
	 * 
	 * @return a summary of the model
	 */
	@Override
	public String toSummaryString() {

		int numInstances = m_data.numInstances();
		Attribute cl = m_data.classAttribute();
		//at.enumerateValues()
		AttributeStats as = m_data.attributeStats(m_data.classIndex());
		int[] classCounts = as.nominalCounts;

		String lineResult = "Classes:\n";
		for(int i = 0; i < classCounts.length; i++) {
			if (i < classCounts.length - 1)
				lineResult = lineResult +  cl.value(i) +", ";
			else
				lineResult = lineResult + cl.value(i) + "\n";
		}
		for(int i = 0; i < classCounts.length; i++) {
			if (i < classCounts.length - 1)
				lineResult = lineResult +  classCounts[i] +", ";
			else
				lineResult = lineResult + classCounts[i] + "\n";
		}
		for(int i = 0; i < classCounts.length; i++) {
			if (i < classCounts.length - 1)
				lineResult = lineResult + Utils.doubleToString((double)classCounts[i]/numInstances*100, 3 + m_numDecimalPlaces, m_numDecimalPlaces) + ", ";
			else
				lineResult = lineResult + Utils.doubleToString((double)classCounts[i]/numInstances*100, 3 + m_numDecimalPlaces, m_numDecimalPlaces) + "\n";
		}
		return lineResult + " \n";
	}


}
