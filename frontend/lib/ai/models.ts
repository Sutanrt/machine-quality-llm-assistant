export const DEFAULT_CHAT_MODEL: string = "chat-model";

export type ChatModel = {
  id: string;
  name: string;
  description: string;
};

export const chatModels: ChatModel[] = [
  
  {
    id: "chat-model-reasoning",
    name: "MCI-Tag Number-Reasoning",
    description:
      "Uses advanced chain-of-thought reasoning for complex problems",
  },{
    id: "chat-model",
    name: "Test 1",
    description: "Advanced multimodal model with vision and text capabilities",
  },

    {
    id: "Machine-Value-reasoning",
    name: "Test 2",
    description:
      "Uses advanced chain-of-thought reasoning for complex problems",
  },
];
