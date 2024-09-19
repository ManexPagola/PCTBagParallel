package weka.classifiers.trees.j48Consolidated;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Class for extending the Instances class in order to add some methods
 * (These methods can be added to the class 'Instances').
 * ************************************************************************
 *
 * @author Jes&uacute;s M. P&eacute;rez (txus.perez@ehu.eus) 
 * @version $Revision: 2.0 $
 */
public class InstancesConsolidated extends Instances {

	/** for serialization */
	private static final long serialVersionUID = 8452710983684965074L;

	/**
	 * Constructor calling the constructor of the superclass
	 * (Not necessary if the above methods are moved to the official class 'Instances')
	 *
	 * @param dataset the set to be copied
	 */
	public InstancesConsolidated(Instances dataset) {
		super(dataset);
	}
	
	/**
	 * Constructor calling the constructor of the superclass
	 * (Not necessary if the above methods are moved to the official class 'Instances')
	 *
	 * @param source the set of instances from which a subset
	 * is to be created
	 * @param first the index of the first instance to be copied
	 * @param toCopy the number of instances to be copied
	 */
	public InstancesConsolidated(Instances source, int first, int toCopy) {
		super(source, first, toCopy);
	}
	
	/**
	 * Gets the vector of classes of the dataset like a set of samples
	 *  
	 * @return the vector of classes
	 */
	public InstancesConsolidated[] getClasses(){
		int numClasses = numClasses();
		InstancesConsolidated[] classesVector = new InstancesConsolidated[numClasses];
		// Sort instances based on the class to extract the set of classes
		sort(classIndex());
		// Determine where each class starts in the sorted dataset
		int[] classIndices = getClassIndices();

		for (int iClass = 0; iClass < numClasses; iClass++) {
			int classSize;
			if (iClass == numClasses - 1) // if the last class
				classSize = numInstances() - classIndices[iClass];
			else
				classSize = classIndices[iClass + 1] - classIndices[iClass]; 
			classesVector[iClass] = new InstancesConsolidated(this, classIndices[iClass], classSize);
		}
		classIndices = null;
		return classesVector;
	}
	
	/**
	 * Creates an index containing the position where each class starts in 
	 * the dataset. The dataset must be sorted by the class attribute.
	 * (based on the method 'createSubsample()' of the class 'SpreadSubsample'
	 *  of the package 'weka.filters.supervised.instances')
	 * 
	 * @return the positions
	 */
	private int[] getClassIndices() {

		// Create an index of where each class value starts
		int [] classIndices = new int [numClasses() + 1];
		int currentClass = 0;
		classIndices[currentClass] = 0;
		for (int i = 0; i < numInstances(); i++) {
			Instance current = instance(i);
			if (current.classIsMissing()) {
				for (int j = currentClass + 1; j < classIndices.length; j++) {
					classIndices[j] = i;
				}
				break;
			} else if (current.classValue() != currentClass) {
				for (int j = currentClass + 1; j <= current.classValue(); j++) {
					classIndices[j] = i;
				}          
				currentClass = (int) current.classValue();
			}
		}
		if (currentClass <= numClasses()) {
			for (int j = currentClass + 1; j < classIndices.length; j++) {
				classIndices[j] = numInstances();
			}
		}
		return classIndices;
	}

	/**
	 * Gets the vector with the size of each class of the dataset
	 *  
	 * @param classesVector the vector of classes of the dataset like a set of samples
	 * @return the vector of classes' size 
	 */
	public int[] getClassesSize(InstancesConsolidated[] classesVector){
		int numClasses = numClasses();
		int classSizeVector[] = new int [numClasses];
		for (int iClass = 0; iClass < numClasses; iClass++)
			classSizeVector[iClass] = classesVector[iClass].numInstances();
		return classSizeVector;
	}

	/**
	 * Adds a set of instances to the end of the set.
	 *
	 * @param instances the set of instances to be added
	 */
	public void add(InstancesConsolidated instances) {
		for(int i = 0; i < instances.numInstances(); i++)
			add(instances.instance(i));
	}
	
	/**
	 * Prints information about the size of the classes and their proportions
	 * and indicates which is the minority class of the sample
	 *
	 * @param dataSize the size of original sample
	 * @param iMinClass the index of the minority class in the original sample
	 * @param classSizeVector the vector with the size of each class of the dataset
	 */
	public void printClassesInformation(int dataSize, int iMinClass, int[] classSizeVector){
		int numClasses = numClasses();
		System.out.println("Minority class value (" + iMinClass +
				"): " + classAttribute().value(iMinClass));
		System.out.println("Classes sizes:");
		for (int iClass = 0; iClass < numClasses; iClass++){
			/** Distribution of the 'iClass'-th class in the original sample */
			float distrClass;
			if (dataSize == 0)
				distrClass = (float)0;
			else
				distrClass = (float)100 * classSizeVector[iClass] / dataSize;
			System.out.print(classSizeVector[iClass] + " (" + Utils.doubleToString(distrClass,2) + "%)");
			if(iClass < numClasses - 1)
				System.out.print(", ");
		}
		System.out.println("");
	}
}
