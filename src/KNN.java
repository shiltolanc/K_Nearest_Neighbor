import java.io.*;
import java.util.*;

public class KNN {

    public static void main(String[] args) {
        if (args.length != 4) {
            System.out.println("Usage: java KNN <training_file> <test_file> <output_file> <K>");
            System.exit(1);
        }

        String trainingFileName = args[0];
        String testFileName = args[1];
        String outputFileName = args[2];
        int K = Integer.parseInt(args[3]);

        ArrayList<DataPoint> trainingData = readData(trainingFileName);
        ArrayList<DataPoint> testData = readData(testFileName);

        int correctPredictions = 0;

        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName));
            writer.write("y,predicted_y");
            for(int i = 1; i <= K; i++){
                writer.write(",distance" + i);
            }
            writer.write("\n");

            // Iterate through test data points
            for (DataPoint testPoint : testData) {
                // Predict the class label for the test point
                int predictedLabel = predictClass(trainingData, testPoint, K);
                // Compare predicted label with the actual label
                if (predictedLabel == testPoint.get_Class()) {
                    correctPredictions++;
                }
                // Write the original label, predicted label, and distances to neighbors to the output file
                StringBuilder neighborDistances = new StringBuilder();
                List<DataPointDistance> distances = getDistances(trainingData, testPoint);
                for (int i = 0; i < Math.min(K, distances.size()); i++) {
                    neighborDistances.append(distances.get(i).distance).append(",");
                }
                writer.write(testPoint.get_Class() + "," + predictedLabel + "," + neighborDistances.toString() + "\n");
            }
            writer.close();
            System.out.println("Output file created successfully: " + outputFileName);
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Calculate accuracy
        double accuracy = (double) correctPredictions / testData.size();
        System.out.println("Accuracy: " + accuracy * 100 + "%");
    }

    private static ArrayList<DataPoint> readData(String fileName) {
        ArrayList<DataPoint> data = new ArrayList<>();
        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader(fileName));
            String line = reader.readLine(); // Skip header
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                double[] attributes = new double[parts.length - 1];
                for (int i = 0; i < attributes.length; i++) {
                    attributes[i] = Double.parseDouble(parts[i]);
                }
                int Class = Integer.parseInt(parts[parts.length - 1]);
                data.add(new DataPoint(attributes, Class));
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    private static double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    private static List<DataPointDistance> getDistances(List<DataPoint> trainingData, DataPoint testPoint) {
        List<DataPointDistance> distances = new ArrayList<>();
        for (DataPoint trainingPoint : trainingData) {
            double distance = euclideanDistance(trainingPoint.getAttributes(), testPoint.getAttributes());
            distances.add(new DataPointDistance(trainingPoint, distance));
        }
        distances.sort(Comparator.comparingDouble(d -> d.distance));
        return distances;
    }

    private static int predictClass(List<DataPoint> trainingData, DataPoint testPoint, int k) {
        Map<Integer, Integer> classVotes = new HashMap<>();
        List<DataPointDistance> distances = getDistances(trainingData, testPoint);
        for (int i = 0; i < k; i++) {
            DataPoint nearestPoint = distances.get(i).point;
            int nearestPointClass = nearestPoint.Class;
            classVotes.put(nearestPointClass, classVotes.getOrDefault(nearestPointClass, 0) + 1);
        }

        // Find the class with the most votes
        int predictedClass = -1;
        int maxVotes = -1;
        for (Map.Entry<Integer, Integer> entry : classVotes.entrySet()) {
            if (entry.getValue() > maxVotes) {
                maxVotes = entry.getValue();
                predictedClass = entry.getKey();
            }
        }

        return predictedClass;
    }

    static class DataPoint {
        private final double[] attributes;
        private final int Class;

        public DataPoint(double[] attributes, int Class) {
            this.attributes = attributes;
            this.Class = Class;
        }

        public double[] getAttributes() {
            return attributes;
        }

        public int get_Class() {
            return Class;
        }

        public String toString() {
            return "Class: " + Class;
        }
    }

    private static class DataPointDistance {
        DataPoint point;
        double distance;

        DataPointDistance(DataPoint point, double distance) {
            this.point = point;
            this.distance = distance;
        }
    }
}