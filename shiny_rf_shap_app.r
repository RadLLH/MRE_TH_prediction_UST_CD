library(shiny)
library(xgboost)
library(fastshap)
library(shapviz)
library(ggplot2)
library(readr)
library(caret)
library(e1071)

# === Load model and reference data ===
model <- readRDS("Model_LASSO_Combined_xgbTree.rds")
data <- read_csv("combined_data_example.csv")
data$M_IUS_TH <- factor(data$M_IUS_TH, levels = c(0, 1), labels = c("Non_TH", "TH"))

X <- as.matrix(sapply(data[, c("Behavior","ESR","X25_hydroxyvitamin_D","BWT","Penetration","Effusion","Contrast_enhancement_extent","ADC")], as.numeric))

# === UI ===
ui <- fluidPage(
  titlePanel("TH26-UST: Web App for Baseline Prediction of Week-26 Transmural Healing in Crohn's Disease"),
  p("This app estimates the probability of achieving transmural healing at week 26 in Crohn's disease patients treated with ustekinumab, based on baseline clinical and MRE features.", style = "color:grey; font-size:small;"),
  
  hr(),
  
  fluidRow(
    column(6, 
      wellPanel(
        h4("Clinical Factors"),
        selectInput("Behavior", "Behavior", choices = c("B1 [inflammatory]" = "1", "B2 [stricturing]" = "2", "B3 [penetrating]" = "3")),
        numericInput("ESR", "Erythrocyte Sedimentation Rate (ESR, mm/h)", value = 20),
        numericInput("X25_hydroxyvitamin_D", "25-hydroxyvitamin D (ng/mL)", value = 15)
      )
    ),
    column(6,
      wellPanel(
        h4("MRE Factors"),
        numericInput("BWT", "Bowel Wall Thickness (BWT, mm)", value = 5),
        selectInput("Penetration", "Penetration", choices = c(
          "No penetration" = "0",
          "Deep ulcer" = "1",
          "Fistula/abscess" = "2"
        )),
        selectInput("Effusion", "Perienteric effusion", choices = c(
          "Similar to the normal mesentery" = "0",
          "Increased mesenteric signal without perienteric effusion" = "1",
          "Increased mesenteric signal with perienteric effusion" = "2"
        )),
        selectInput("Contrast_enhancement_extent", "Mural enhancement degree", choices = c(
          "Similar to the normal bowel" = "0",
          "Stronger than that of normal bowel wall but weaker than that of adjacent vessels" = "1",
          "Close to the adjacent blood vessels" = "2"
        )),
        numericInput("ADC", HTML("Apparent Diffusion Coefficient (ADC, x10<sup>-3</sup> mm<sup>2</sup>/s)"), value = 1.0)
      )
    )
  ),
  
  fluidRow(
    column(12, align="center",
      actionButton("predict", "Predict", style = "width: 200px; margin-bottom: 20px;")
    )
  ),

  hr(),
  
  fluidRow(
    column(12,
      h4("Prediction Result"),
      verbatimTextOutput("result"),
      h4("SHAP Waterfall Plot"),
      plotOutput("waterfall_plot", height = "600px")
    )
  )
)

# === Server ===
server <- function(input, output) {
  input_data <- eventReactive(input$predict, {
    data.frame(
      Behavior = as.numeric(input$Behavior),
      ESR = as.numeric(input$ESR),
      X25_hydroxyvitamin_D = as.numeric(input$X25_hydroxyvitamin_D),
      BWT = as.numeric(input$BWT),
      Penetration = as.numeric(input$Penetration),
      Effusion = as.numeric(input$Effusion),
      Contrast_enhancement_extent = as.numeric(input$Contrast_enhancement_extent),
      ADC = as.numeric(input$ADC)
    )
  })

  shap_single <- eventReactive(input$predict, {
    set.seed(160)
    newx <- input_data()
    shap_val <- fastshap::explain(
      model,
      feature_names = colnames(X),
      X = X,
      newdata = as.matrix(newx),
      pred_wrapper = function(object, newdata) predict(object, newdata = newdata, type = "prob")[, "TH"],
      nsim = 100
    )
    # fastshap::explain can return a vector for a single prediction.
    # To prevent errors downstream when using colnames, we ensure it's a matrix.
    if (!is.matrix(shap_val)) {
      shap_val <- matrix(shap_val, nrow = 1, dimnames = list(NULL, names(shap_val)))
    }

    # Use a named vector for renaming
    name_map <- c(
      "Contrast_enhancement_extent" = "Mural enhancement degree",
      "X25_hydroxyvitamin_D" = "25-hydroxyvitamin D",
      "Effusion" = "Perienteric effusion"
    )

    # Rename columns of shap_val
    shap_cn <- colnames(shap_val)
    shap_remap_idx <- which(shap_cn %in% names(name_map))
    if (length(shap_remap_idx) > 0) {
      shap_cn[shap_remap_idx] <- name_map[shap_cn[shap_remap_idx]]
      colnames(shap_val) <- shap_cn
    }

    # Rename columns of newx
    newx_cn <- colnames(newx)
    newx_remap_idx <- which(newx_cn %in% names(name_map))
    if (length(newx_remap_idx) > 0) {
      newx_cn[newx_remap_idx] <- name_map[newx_cn[newx_remap_idx]]
      colnames(newx) <- newx_cn
    }

    shapviz(shap_val, X = newx)
  })

  output$result <- renderPrint({
    req(input_data())
    prob <- predict(model, newdata = as.matrix(input_data()), type = "prob")[, "TH"]
    paste0("Predicted probability of TH: ", round(prob, 3))
  })

  output$waterfall_plot <- renderPlot({
    sv <- shap_single()
    req(sv)
    
    p <- sv_waterfall(sv, max_display = 8) + theme_bw(base_size = 18)

    # The E[f(x)] and f(x) annotations are in the last two layers。
    # 增加它们的字体大小，默认值较小，尝试设置为 5。
    p$layers[[length(p$layers)]]$aes_params$size <- 5
    p$layers[[length(p$layers) - 1]]$aes_params$size <- 5
    
    p
  })
}

shinyApp(ui = ui, server = server)
