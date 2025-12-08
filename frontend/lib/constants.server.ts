// lib/constants.server.ts  (SERVER ONLY; boleh Node API)
import { generateDummyPassword } from "./db/utils";

export const DUMMY_PASSWORD = generateDummyPassword();
