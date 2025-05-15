import { createContext, useState } from "react";
import { run } from "../config/server";

export const Context = createContext();

const ContextProvider = (props) => {
  const [input, setInput] = useState("");
  const [resultData, setResultData] = useState("");
  const [loading, setLoading] = useState(false);

  const onSent = async (prompt) => {
    setLoading(true);
    const response = await run(prompt || input);
    setResultData(response);
    setLoading(false);
    setInput("");
  };

  const contextValue = {
    input,
    setInput,
    resultData,
    onSent,
    loading,
  };

  return (
    <Context.Provider value={contextValue}>
      {props.children}
    </Context.Provider>
  );
};

export default ContextProvider;
