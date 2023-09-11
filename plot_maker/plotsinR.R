library(jsonlite)
library(ggplot2)
# Create a dictionary (list of lists) containing the model accuracies
models <- list(
  Model1 = c(0.1, 0.2, 0.3, 0.4, 0.5),
  Model2 = c(0.15, 0.25, 0.35, 0.45, 0.55),
  Model3 = c(0.08, 0.18, 0.28, 0.38, 0.48)
)

# Convert the dictionary to a data frame
df <- data.frame(
  Epoch = 1:length(models$Model1),
  Model1 = models$Model1,
  Model2 = models$Model2,
  Model3 = models$Model3
)

# Melt the data frame to long format for ggplot
df_long <- reshape2::melt(df, id.vars = "Epoch")

# Create the plot using ggplot2
ggplot(data = df_long, aes(x = Epoch, y = value, color = variable)) +
  geom_line() +
  labs(
    title = "Model Accuracies vs. Epoch",
    x = "Epoch",
    y = "Accuracy",
    color = "Model"
  ) +
  theme_minimal()
# Load the JSON file
json_data <- fromJSON("/home/karan/Documents/GitHub/BTP/data_convertor/exp_metrics.json")
print(json_data['epoch'])
# Extract data from the JSON
# data <- json_data$data

# Convert the data to a data frame
# df <- as.data.frame(data)

# Create a scatter plot using ggplot2
# ggplot(df, aes(x = x, y = y)) +
#   geom_point() +
#   labs(title = "Scatter Plot of JSON Data", x = "X-axis", y = "Y-axis")
