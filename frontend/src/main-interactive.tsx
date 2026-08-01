import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import { InteractiveApp } from "./InteractiveApp.tsx";

const rootElement = document.getElementById("root");
if (!rootElement) throw new Error("Root element #root was not found");

createRoot(rootElement).render(
  <StrictMode>
    <InteractiveApp />
  </StrictMode>,
);
