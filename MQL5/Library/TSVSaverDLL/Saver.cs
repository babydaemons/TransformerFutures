using System;
using System.IO;
using System.IO.Compression;

/// <summary>
/// File: Saver.cs
/// 
/// ソースコードの役割:
/// MetaTrader 5 (MQL5) からDLLインポート経由で呼び出され、
/// ティックデータ（Bid/Ask）を週次のTSVファイルとして保存します。
/// ファイルクローズ時にはディスク容量節約のため自動的にGZip圧縮を行います。
/// MT5のストラテジーテスターにおけるサンドボックス制限（ファイル作成エラー5004等）を
/// 回避し、OSの任意のディレクトリに階層フォルダを安全に構築します。
/// </summary>
public class Saver
{
    // ----------------------------------------------------------------
    // 保存先のベースディレクトリ
    // ※MT5の制限を受けない場所（例: Cドライブ直下の専用フォルダ）を指定します
    // ----------------------------------------------------------------
    private static readonly string BaseDirectory = @"C:\TransformerFutures\data\original\MT5";

    // スレッドセーフ用ロックオブジェクト
    private static readonly object _lock = new object();

    // パフォーマンス向上のため、開いているファイルハンドルを保持
    private static string _currentFilePath = string.Empty;
    private static StreamWriter _writer = null;

    /// <summary>
    /// ティックデータをTSVに保存（週次でファイルを自動切り替え）
    /// </summary>
    /// <param name="symbol">通貨ペア名 (例: USDJPY)</param>
    /// <param name="timestamp">ティックの発生日時</param>
    /// <param name="ask">Askレート</param>
    /// <param name="bid">Bidレート</param>
    public static void Save(string symbol, string timestamp, double ask, double bid)
    {
        lock (_lock)
        {
            try
            {
                // 文字列からDateTimeに復元 (ミリ秒も保持されます)
                DateTime date = DateTime.Parse(timestamp);

                // 1. 該当週の月曜日の日付を計算
                int diff = (7 + (date.DayOfWeek - DayOfWeek.Monday)) % 7;
                DateTime monday = date.AddDays(-1 * diff).Date;

                string year_str = monday.ToString("yyyy");
                string week_str = monday.ToString("yyyyMMdd");

                // 2. ディレクトリパスの構築 (例: C:\transformer_futures_data\USDJPY\2026)
                string dir_path = Path.Combine(BaseDirectory, symbol, year_str);

                // ディレクトリが存在しない場合は一括で作成（C#ならエラーにならない）
                if (!Directory.Exists(dir_path))
                {
                    Directory.CreateDirectory(dir_path);
                }

                // 3. ファイルパスの構築 (例: C:\transformer_futures_data\USDJPY\2026\USDJPY-20260302.tsv)
                string file_path = Path.Combine(dir_path, $"{symbol}-{week_str}.tsv");

                // 4. 週が変わった、または初めての書き込みの場合のハンドリング
                if (_currentFilePath != file_path)
                {
                    // 既存の開いているファイルがあれば閉じて圧縮する
                    if (_writer != null)
                    {
                        // 内部で_writerのクローズとGZip圧縮が行われる
                        Close();
                    }

                    bool is_new_file = !File.Exists(file_path) && !File.Exists(file_path + ".gz");

                    // ファイルを追記モードで開く
                    _writer = new StreamWriter(file_path, append: true);

                    // 新規作成時のみヘッダーを書き込む
                    if (is_new_file)
                    {
                        _writer.WriteLine("timestamp\tbid\task");
                    }

                    _currentFilePath = file_path;
                }

                // 5. データベースフレンドリーな形式 (YYYY-MM-DD HH:MM:SS.mmm) にフォーマットして書き込み
                string time_str = date.ToString("yyyy-MM-dd HH:mm:ss.fff");
                _writer.WriteLine($"{time_str}\t{bid}\t{ask}");

            }
            catch (Exception)
            {
                // DLL内でUnhandled Exceptionが発生するとMT5ごとクラッシュするため、
                // 最低限のエラーは握りつぶすか、別ファイルにログ出力させます。
            }
        }
    }

    /// <summary>
    /// MT5の OnDeinit() から呼び出して、ファイルを安全に閉じ、GZip圧縮を行うメソッド
    /// </summary>
    public static void Close()
    {
        lock (_lock)
        {
            if (_writer != null)
            {
                _writer.Flush();
                _writer.Close();
                _writer.Dispose();
                _writer = null;

                // GZip圧縮処理の実行
                if (!string.IsNullOrEmpty(_currentFilePath) && File.Exists(_currentFilePath))
                {
                    string gzip_file_path = _currentFilePath + ".gz";

                    try
                    {
                        using (FileStream original_file_stream = new FileStream(_currentFilePath, FileMode.Open, FileAccess.Read))
                        using (FileStream compressed_file_stream = new FileStream(gzip_file_path, FileMode.Create))
                        using (GZipStream compression_stream = new GZipStream(compressed_file_stream, CompressionMode.Compress))
                        {
                            original_file_stream.CopyTo(compression_stream);
                        }

                        // 圧縮完了後に元の非圧縮TSVファイルを削除
                        File.Delete(_currentFilePath);
                    }
                    catch (Exception)
                    {
                        // 圧縮エラー時のハンドリング（MT5クラッシュ防止のため握りつぶす）
                    }
                }

                _currentFilePath = string.Empty;
            }
        }
    }
}
