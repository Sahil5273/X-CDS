import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { ReportApp } from "./ReportApp.tsx";

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Root element #root was not found");

createRoot(rootElement).render(
  <StrictMode>
    <ReportApp />
  </StrictMode>,
);
