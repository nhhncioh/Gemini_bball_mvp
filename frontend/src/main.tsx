import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";

// 🔑  Import *after* App so code-splitting still works
import { worker } from "./mocks/browser";

async function bootstrap() {
  // Start MSW (dev only) and wait for it to be ready
  if (import.meta.env.DEV) {
    await worker.start({
      onUnhandledRequest: "bypass",           // let non-API requests pass through
    });
  }

  // Now render the React app
  ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  );
}

bootstrap();
