from autogluon.bench.utils.general_utils import download_dir_from_s3, download_file_from_s3, upload_to_s3


def test_upload_to_s3(mocker, tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.touch()

    s3_client_mock = mocker.Mock()
    mocker.patch("boto3.client", return_value=s3_client_mock)

    bucket = "test_bucket"
    benchmark_name = "test_benchmark"

    s3_path = upload_to_s3(s3_bucket=bucket, s3_dir=benchmark_name, local_path=str(config_file))

    s3_client_mock.upload_file.assert_called_once_with(
        str(config_file),
        bucket,
        f"{benchmark_name}/config.yaml",
    )

    assert s3_path == f"s3://{bucket}/{benchmark_name}/config.yaml"


def test_download_file_from_s3(mocker, tmp_path):
    s3_client_mock = mocker.Mock()
    mocker.patch("boto3.client", return_value=s3_client_mock)

    s3_path = "s3://test_bucket/configs/test_file.txt"
    local_path = download_file_from_s3(s3_path, str(tmp_path))

    s3_client_mock.download_file.assert_called_once_with(
        "test_bucket",
        "configs/test_file.txt",
        str(tmp_path / "test_file.txt"),
    )

    assert local_path == str(tmp_path / "test_file.txt")


def test_download_dir_from_s3(mocker, tmp_path):
    s3_client_mock = mocker.Mock()
    mocker.patch("boto3.client", return_value=s3_client_mock)

    # Mock the response of list_objects
    mock_response = {
        "Contents": [
            {"Key": "configs/test_file1.txt"},
            {"Key": "configs/test_file2.txt"},
        ]
    }
    s3_client_mock.list_objects.return_value = mock_response

    s3_path = "s3://test_bucket/configs"
    local_path = download_dir_from_s3(s3_path, str(tmp_path))

    expected_calls = [
        (("test_bucket", "configs/test_file1.txt", str(tmp_path / "test_file1.txt")),),
        (("test_bucket", "configs/test_file2.txt", str(tmp_path / "test_file2.txt")),),
    ]
    assert s3_client_mock.download_file.call_args_list == expected_calls

    assert local_path == str(tmp_path)
